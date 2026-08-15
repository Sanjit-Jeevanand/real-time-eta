from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eta.data.weather import WEATHER_COLUMNS
from eta.features.families import register_all
from eta.features.registry import REGISTRY
from eta.serving.app import build_app
from eta.serving.predictor import Predictor, StageTimer
from eta.serving.store import FeatureStore

register_all()

ALPHAS = (0.05, 0.5, 0.75, 0.9, 0.95)
WHEN = dt.datetime(2023, 6, 1, 9, 0, tzinfo=dt.UTC)


class _Boom:
    """A store backend that always fails, the way a dead Redis does."""

    def mget(self, keys: list[str]) -> list[float | None]:
        msg = "connection refused"
        raise ConnectionError(msg)


class _Dict:
    def __init__(self, data: dict[str, float] | None = None) -> None:
        self.data = data or {}
        self.calls = 0

    def mget(self, keys: list[str]) -> list[float | None]:
        self.calls += 1
        return [self.data.get(k) for k in keys]


def _predictor(backend: Any, q: float = 120.0) -> Predictor:
    names = REGISTRY.names
    route = {
        "route_distance_m": 8_000.0,
        "free_flow_duration_s": 800.0,
        "turn_count": 12.0,
        "highway_fraction": 0.3,
        "is_intra_zone": 0.0,
        "structural_detour_ratio": 1.3,
        "detour_trips": 500.0,
    }
    zone: dict[str, Any] = {
        "area_km2": 2.0,
        "is_airport": 0.0,
        "borough_id": 1.0,
        "hist_mean_duration_s": 900.0,
        "reference_speed_ms": 8.0,
        "_embedding": [0.1] * 16,
    }
    return Predictor(
        alphas=ALPHAS,
        feature_names=list(names),
        route_lookup={(1, 2): route},
        zone_lookup={1: dict(zone), 2: dict(zone)},
        store=FeatureStore(backend=backend),
        # Stand-in for the Treelite-compiled boosters: ordered, deterministic.
        booster_predict=lambda x: np.array([[300.0, 600.0, 800.0, 1000.0, 1200.0]]),
        conformal_q=q,
        lo_idx=0,
        hi_idx=4,
        weather_defaults=dict.fromkeys(WEATHER_COLUMNS, 0.0),
    )


# ------------------------------------------------------------ degradation ---
def test_a_dead_store_degrades_instead_of_failing() -> None:
    """The contract: Redis dying must never fail a request."""
    pred = _predictor(_Boom()).predict(1, 2, WHEN)
    assert pred.degraded is True
    assert pred.quantiles["p75"] == 800.0


def test_degradation_reports_missing_not_zero() -> None:
    """Zero congestion is a confident lie; missing is the truth the model was trained on."""
    store = FeatureStore(backend=_Boom())
    result = store.fetch(["cong:distance:1:15m", "cong:duration:1:15m"])
    assert result.degraded is True
    assert set(result.values.values()) == {None}, "a failed fetch must not fabricate zeros"


def test_a_healthy_store_with_no_state_yet_is_not_degraded() -> None:
    """Cold start is a legitimate miss, not a failure -- they must not be conflated."""
    result = FeatureStore(backend=_Dict()).fetch(["cong:distance:9:15m"])
    assert result.degraded is False
    assert result.values["cong:distance:9:15m"] is None


def test_failures_can_be_made_fatal_when_that_is_wanted() -> None:
    store = FeatureStore(backend=_Boom(), degrade_on_failure=False)
    with pytest.raises(ConnectionError):
        store.fetch(["cong:distance:1:15m"])


def test_the_store_is_hit_once_per_request_not_once_per_feature() -> None:
    backend = _Dict()
    _predictor(backend).predict(1, 2, WHEN)
    assert backend.calls == 1, "congestion features must share one pipelined round trip"


def test_redis_keys_come_from_the_registry() -> None:
    """A new live feature must not need a second, hand-maintained list."""
    keys = _predictor(_Dict())._redis_keys(7)
    assert len(keys) == len(REGISTRY.required_redis_keys())

    congestion = [k for k in keys if k.startswith("cong:")]
    weather = [k for k in keys if k.startswith("wx:")]
    assert congestion and weather, "both key families must be fetched"
    assert all(":7:" in k for k in congestion), "congestion keys bind to the requested zone"
    assert all(":7:" not in k for k in weather), "weather is global, not per-zone"


def test_fetched_weather_reaches_the_features_instead_of_being_discarded() -> None:
    """Routing wx: keys to the congestion namespace fails silently -- every weather
    feature would read NaN and the service would still return a plausible answer."""
    backend = _Dict({f"wx:{c}": 7.5 for c in WEATHER_COLUMNS})
    fetched: dict[str, float | None] = {f"wx:{c}": 7.5 for c in WEATHER_COLUMNS}
    fetched["cong:distance:7:15m"] = 100.0
    congestion, weather = _predictor(backend)._split(fetched)
    assert weather == dict.fromkeys(WEATHER_COLUMNS, 7.5), "wx: prefix must be stripped"
    assert congestion == {"cong:distance:7:15m": 100.0}, "congestion keeps its full key"


# ---------------------------------------------------------------- timings ---
def test_every_stage_is_timed_separately() -> None:
    pred = _predictor(_Dict()).predict(1, 2, WHEN)
    assert set(pred.timings) == {
        "route_lookup",
        "store_fetch",
        "feature_assembly",
        "inference",
        "conformal",
    }
    assert pred.total_ms == pytest.approx(sum(pred.timings.values()), abs=0.01)


def test_stage_timer_attributes_elapsed_time_to_the_right_stage() -> None:
    timer = StageTimer()
    with timer:
        timer.mark("fast")
        for _ in range(200_000):
            pass
        timer.mark("slow")
    assert timer.stages["slow"] > timer.stages["fast"]


# ------------------------------------------------------------------ API -----
def test_the_response_carries_every_quantile_and_the_interval() -> None:
    app = build_app(_predictor(_Dict()), served_quantile=0.75, target_coverage=0.90)
    client = TestClient(app)
    r = client.post("/eta", json={"pickup_zone": 1, "dropoff_zone": 2})

    assert r.status_code == 200
    body = r.json()
    # A single number would hide the uncertainty this project exists to quantify.
    assert set(body["quantiles"]) == {"p5", "p50", "p75", "p90", "p95"}
    assert body["served"] == body["quantiles"]["p75"]
    assert body["served_quantile"] == 0.75
    assert body["interval_low"] < body["quantiles"]["p5"]
    assert body["interval_high"] > body["quantiles"]["p95"]


def test_stage_timings_ship_on_the_response_header() -> None:
    app = build_app(_predictor(_Dict()), served_quantile=0.75, target_coverage=0.90)
    r = TestClient(app).post("/eta", json={"pickup_zone": 1, "dropoff_zone": 2})
    header = r.headers["X-Stage-Timings"]
    for stage in ("route_lookup", "store_fetch", "feature_assembly", "inference"):
        assert stage in header


def test_a_degraded_request_still_returns_200_and_says_so() -> None:
    app = build_app(_predictor(_Boom()), served_quantile=0.75, target_coverage=0.90)
    r = TestClient(app).post("/eta", json={"pickup_zone": 1, "dropoff_zone": 2})

    assert r.status_code == 200, "a store outage must not surface as an error"
    assert r.json()["degraded"] is True
    assert r.headers["X-Degraded"] == "true"


def test_the_conformal_interval_widens_by_exactly_q() -> None:
    pred = _predictor(_Dict(), q=250.0).predict(1, 2, WHEN)
    assert pred.interval[0] == pytest.approx(300.0 - 250.0)
    assert pred.interval[1] == pytest.approx(1200.0 + 250.0)


def test_a_crossed_booster_cannot_produce_a_promise_below_the_median() -> None:
    """Serve-time ordering: the last line of defence if a model regresses."""
    p = _predictor(_Dict())
    p.booster_predict = lambda x: np.array([[300.0, 900.0, 700.0, 1000.0, 1200.0]])
    pred = p.predict(1, 2, WHEN)
    assert pred.quantiles["p75"] >= pred.quantiles["p50"]


def test_health_reports_the_served_quantile() -> None:
    app = build_app(_predictor(_Dict()), served_quantile=0.75, target_coverage=0.90)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert body["served_quantile"] == 0.75


def test_the_serving_path_does_not_import_torch() -> None:
    """The OpenMP guard preloads torch, but only for training.

    Serving must not inherit that: torch costs seconds of startup and hundreds of MB
    of resident memory, on a box budgeted at 2 vCPU and 4GB. Run in a subprocess
    because torch may already be loaded by the training tests in this session.
    """
    import os
    import pathlib
    import subprocess
    import sys

    import eta

    code = (
        "import sys\n"
        "import eta.serving\n"
        "from eta.serving.app import build_app\n"
        "from eta.serving.predictor import Predictor\n"
        "assert 'torch' not in sys.modules, 'serving imported torch'\n"
        "print('clean')\n"
    )
    src = str(pathlib.Path(eta.__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": src},
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "clean" in proc.stdout

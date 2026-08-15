"""Per-stage latency measurement against the real artifacts.

Measures the request path, not a synthetic loop: real zone-pair route matrix, real
zone table, real feature registry, real boosters. The store is a local dict rather
than a network Redis, so the `store_fetch` row here is a **floor**, not a
prediction -- it measures everything except the network hop. That is stated in the
output rather than left for a reader to assume.
"""

from __future__ import annotations

import json
import statistics as st
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from eta.features.context import congestion_key, load_static_tables
from eta.features.families import register_all
from eta.features.registry import REGISTRY
from eta.logging import get_logger
from eta.models.dataset import feature_names

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from eta.config import Settings

__all__ = ["BenchResult", "run_bench"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BenchResult:
    stage: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


class _WarmDict:
    """Stands in for Redis: same key set, same one-shot fetch, no network."""

    def __init__(self, data: dict[str, float]) -> None:
        self.data = data

    def mget(self, keys: list[str]) -> list[float | None]:
        return [self.data.get(k) for k in keys]


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p))


def run_bench(
    settings: Settings,
    reports: Path,
    requests: int = 2_000,
    seed: int = 0,
    use_redis: bool = False,
    use_treelite: bool = False,
) -> dict[str, Any]:
    import lightgbm as lgb

    from eta.data.splits import Split
    from eta.data.weather import WEATHER_COLUMNS
    from eta.models.dataset import TARGET, load_split
    from eta.serving.predictor import Predictor
    from eta.serving.store import FeatureStore

    register_all()
    paths = settings.paths.resolve()
    processed = paths["processed_dir"]
    static = load_static_tables(processed)
    names = feature_names()
    alphas = tuple(settings.model.quantiles)

    matrix_path = processed / "model_matrix_tune.parquet"
    train = load_split(matrix_path, Split.TRAIN).head(400_000)
    x = train.select(names).to_numpy()
    y = train[TARGET].to_numpy().astype(np.float64)

    # Small boosters: this measures the *serving* path, and inference cost scales
    # with trees, so the row below is honestly labelled as a floor for 600-round models.
    boosters = []
    for a in alphas:
        boosters.append(
            lgb.train(
                {
                    "objective": "quantile",
                    "alpha": float(a),
                    "num_leaves": 63,
                    "verbosity": -1,
                    "seed": seed,
                    "num_threads": 1,
                },
                lgb.Dataset(x, label=y, feature_name=list(names)),
                num_boost_round=120,
            )
        )

    def predict_all(row: np.ndarray) -> np.ndarray:
        return np.column_stack([b.predict(row) for b in boosters])

    predict_fn: Any = predict_all
    if use_treelite:
        from eta.serving.compile import compile_boosters

        predict_fn = compile_boosters(
            boosters, settings.paths.resolve()["artifacts_dir"] / "treelite"
        )

    rng = np.random.default_rng(seed)
    pairs = list(static.matrix.select("pu_zone", "do_zone").head(5_000).iter_rows())
    warm: dict[str, float] = {}
    for pu, _ in pairs[:2_000]:
        for kind, val in (("distance", 40_000.0), ("duration", 4_000.0), ("completed", 25.0)):
            for w in (15, 30, 60):
                warm[congestion_key(kind, int(pu), w)] = val
    warm.update({f"wx:{c}": 10.0 for c in WEATHER_COLUMNS})

    backend: Any = _WarmDict(warm)
    if use_redis:
        from eta.serving.store import RedisStore

        redis_store = RedisStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            socket_timeout_s=settings.redis.socket_timeout_s,
        ).connect()
        redis_store.mset(warm)
        backend = redis_store

    predictor = Predictor(
        alphas=alphas,
        feature_names=names,
        route_lookup=static.route_lookup(),
        zone_lookup=static.zone_lookup(),
        store=FeatureStore(backend=backend),
        booster_predict=predict_fn,
        conformal_q=180.0,
        lo_idx=0,
        hi_idx=len(alphas) - 1,
        weather_defaults=dict.fromkeys(WEATHER_COLUMNS, 10.0),
    )

    import datetime as dt

    base = dt.datetime(2023, 6, 15, 17, 30, tzinfo=dt.UTC)
    stages: dict[str, list[float]] = {}
    totals: list[float] = []

    for i in range(requests):
        pu, do = pairs[int(rng.integers(0, len(pairs)))]
        pred = predictor.predict(int(pu), int(do), base + dt.timedelta(seconds=i))
        for stage, ms in pred.timings.items():
            stages.setdefault(stage, []).append(ms)
        totals.append(pred.total_ms)

    results = [
        BenchResult(
            stage=stage,
            p50_ms=_percentile(vals, 50),
            p95_ms=_percentile(vals, 95),
            p99_ms=_percentile(vals, 99),
            mean_ms=st.fmean(vals),
        )
        for stage, vals in stages.items()
    ]
    results.append(
        BenchResult(
            stage="TOTAL",
            p50_ms=_percentile(totals, 50),
            p95_ms=_percentile(totals, 95),
            p99_ms=_percentile(totals, 99),
            mean_ms=st.fmean(totals),
        )
    )

    store_label = "real Redis over TCP" if use_redis else "local dict (no network)"
    infer_label = "Treelite-compiled" if use_treelite else "LightGBM predict, 120 rounds"
    lines = [
        f"## Per-stage latency -- store: {store_label}; inference: {infer_label}",
        "",
        "| stage | p50 | p95 | p99 | mean |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r.stage} | {r.p50_ms:.3f}ms | {r.p95_ms:.3f}ms | {r.p99_ms:.3f}ms | {r.mean_ms:.3f}ms |"
        for r in results
    ]
    lines += [
        "",
        f"{requests:,} requests, {len(names)} features, {len(alphas)} quantile models, "
        f"{len(REGISTRY.required_redis_keys())} store keys per request.",
        "",
        (
            "`store_fetch` crosses a **real TCP connection** to Redis on this host. "
            "That is a loopback hop, not a datacentre one, so it remains a floor -- but "
            "it now includes protocol encoding, syscalls and the server round trip."
            if use_redis
            else "**`store_fetch` is a local dict, so it is a floor, not a forecast** -- "
            "it measures serialisation and dispatch with the network hop removed."
        ),
        (
            "Inference is **Treelite-compiled**, which is the path that would actually serve."
            if use_treelite
            else "Inference uses 120-round boosters and is not Treelite-compiled, so that "
            "row is a ceiling that compilation is expected to lower."
        ),
        "",
        "Measured on an M4, not the 2-vCPU target box, so the budget is not discharged "
        "by this table.",
    ]

    # Distinct filename per configuration -- two runs with different backends are
    # different measurements and must not overwrite each other.
    suffix = f"{'redis' if use_redis else 'dict'}_{'treelite' if use_treelite else 'lgbm'}"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"latency_{suffix}.md").write_text("\n".join(lines) + "\n")
    (reports / f"latency_{suffix}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n"
    )
    log.info(
        "bench_complete",
        requests=requests,
        p50_ms=round(_percentile(totals, 50), 3),
        p99_ms=round(_percentile(totals, 99), 3),
        target_ms=settings.serving.p99_target_ms,
        report=f"latency_{suffix}.md",
    )
    return {"results": [asdict(r) for r in results], "markdown": "\n".join(lines)}


def _train_boosters(settings: Settings, seed: int = 0) -> tuple[list[Any], list[str], Any]:
    """Shared setup: the same five quantile boosters the bench and compiler both need."""
    import lightgbm as lgb

    from eta.data.splits import Split
    from eta.models.dataset import TARGET, load_split

    register_all()
    processed = settings.paths.resolve()["processed_dir"]
    names = feature_names()
    alphas = tuple(settings.model.quantiles)
    train = load_split(processed / "model_matrix_tune.parquet", Split.TRAIN).head(400_000)
    x = train.select(names).to_numpy()
    y = train[TARGET].to_numpy().astype(np.float64)

    boosters = [
        lgb.train(
            {
                "objective": "quantile",
                "alpha": float(a),
                "num_leaves": 63,
                "verbosity": -1,
                "seed": seed,
                "num_threads": 1,
            },
            lgb.Dataset(x, label=y, feature_name=list(names)),
            num_boost_round=120,
        )
        for a in alphas
    ]
    return boosters, names, x


def run_compile_comparison(settings: Settings, reports: Path) -> dict[str, Any]:
    """Treelite-compile the boosters, verify they agree, then time both."""
    from eta.serving.compile import compare_predictions, compile_boosters, time_single_row

    boosters, names, x = _train_boosters(settings)
    artifacts = settings.paths.resolve()["artifacts_dir"] / "treelite"
    compiled = compile_boosters(boosters, artifacts)

    check = np.ascontiguousarray(x[:2_000], dtype=np.float64)
    raw_batch = np.column_stack([b.predict(check) for b in boosters])
    compiled_batch = compiled(check)
    agreement = compare_predictions(raw_batch, compiled_batch)

    row = np.ascontiguousarray(x[:1], dtype=np.float64)

    def raw_predict(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack([b.predict(r) for b in boosters])

    raw_t = time_single_row(raw_predict, row)
    compiled_t = time_single_row(compiled, row)
    speedup = raw_t["p50_ms"] / compiled_t["p50_ms"] if compiled_t["p50_ms"] else float("nan")

    lines = [
        "## Treelite compilation",
        "",
        "| implementation | p50 | p95 | p99 | mean |",
        "|---|---|---|---|---|",
        f"| LightGBM `predict` | {raw_t['p50_ms']:.3f}ms | {raw_t['p95_ms']:.3f}ms | "
        f"{raw_t['p99_ms']:.3f}ms | {raw_t['mean_ms']:.3f}ms |",
        f"| Treelite compiled | {compiled_t['p50_ms']:.3f}ms | {compiled_t['p95_ms']:.3f}ms | "
        f"{compiled_t['p99_ms']:.3f}ms | {compiled_t['mean_ms']:.3f}ms |",
        "",
        f"**Speedup at p50: {speedup:.2f}x** on single-row inference, "
        f"{len(boosters)} models, {len(names)} features.",
        "",
        "### Agreement",
        "",
        f"- max absolute difference: **{agreement['max_abs_diff']:.6f}s**",
        f"- mean absolute difference: {agreement['mean_abs_diff']:.8f}s",
        f"- rows within 1e-4s: **{agreement['within_tolerance']:.2%}**",
        "",
        "Checked before the timing, because a faster model that disagrees with the "
        "trained one is not an optimisation.",
    ]
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "treelite.md").write_text("\n".join(lines) + "\n")
    (reports / "treelite.json").write_text(
        json.dumps({"raw": raw_t, "compiled": compiled_t, "agreement": agreement}, indent=2) + "\n"
    )
    log.info(
        "compile_comparison",
        speedup=round(speedup, 3),
        raw_p50_ms=round(raw_t["p50_ms"], 4),
        compiled_p50_ms=round(compiled_t["p50_ms"], 4),
        max_abs_diff=agreement["max_abs_diff"],
    )
    return {"markdown": "\n".join(lines), "speedup": speedup, "agreement": agreement}

"""Phase 6 step 6: SHAP on the served P75 model, sliced by segment.

Sliced, because a single global importance ranking answers a question nobody asked.
The claim worth testing is *conditional*: that congestion features dominate at PM
peak and route features dominate off-peak. A global bar chart cannot confirm or
refute that; per-segment attribution can.

SHAP values come from LightGBM's exact TreeSHAP (`pred_contrib=True`), not the
`shap` package's sampling approximations -- for a tree ensemble the exact values are
cheap and there is no reason to estimate them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from eta.features.registry import REGISTRY
from eta.logging import get_logger
from eta.models.quantile.lgbm import LgbmQuantile, default_params

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from eta.config import Settings

__all__ = ["SegmentAttribution", "run_explain_phase"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SegmentAttribution:
    axis: str
    bucket: str
    rows: int
    family_share: dict[str, float]
    top_features: list[tuple[str, float]]


def _family_of(name: str) -> str:
    try:
        return str(REGISTRY[name].family.value)
    except KeyError:  # pragma: no cover - defensive
        return "unknown"


def run_explain_phase(
    settings: Settings,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: Sequence[str],
    reports: Path,
    sample: int = 60_000,
) -> dict[str, Any]:
    q_star = float(settings.cost.optimal_quantile)
    path = reports / "quantile_summary.json"
    params = default_params()
    if path.exists():
        tuned = json.loads(path.read_text()).get("tuned_params", {})
        params = {float(k): v for k, v in tuned.items()}.get(q_star, default_params())

    model = LgbmQuantile(alpha=q_star, features=features, params=params)
    model.fit(train, seed=settings.model.seeds[0])

    rng = np.random.default_rng(0)
    idx = rng.choice(test.height, size=min(sample, test.height), replace=False)
    subset = test[idx]
    x = subset.select(features).to_numpy()

    # (n_rows, n_features + 1); the trailing column is the base value.
    contrib: NDArray[np.float64] = np.asarray(
        model._booster.predict(x, pred_contrib=True), dtype=np.float64
    )
    shap = np.abs(contrib[:, :-1])
    families = [_family_of(f) for f in features]

    log.info(
        "shap_computed",
        rows=int(shap.shape[0]),
        features=len(features),
        alpha=q_star,
        base_value=round(float(contrib[0, -1]), 1),
    )

    results: list[SegmentAttribution] = []
    for axis in ("seg_time", "seg_trip_length", "seg_zone_density", "seg_weather"):
        if axis not in subset.columns:
            continue
        values = subset[axis].to_numpy()
        for bucket in np.unique(values):
            if bucket is None:
                continue
            mask = values == bucket
            if int(mask.sum()) < 1_000:
                continue
            mean_abs = shap[mask].mean(axis=0)
            total = float(mean_abs.sum()) or 1.0

            by_family: dict[str, float] = {}
            for fam, value in zip(families, mean_abs, strict=True):
                by_family[fam] = by_family.get(fam, 0.0) + float(value)
            order = np.argsort(mean_abs)[::-1][:5]

            results.append(
                SegmentAttribution(
                    axis=axis,
                    bucket=str(bucket),
                    rows=int(mask.sum()),
                    family_share={k: v / total for k, v in sorted(by_family.items())},
                    top_features=[(features[i], float(mean_abs[i])) for i in order],
                )
            )

    summary = _summarise(results)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "shap.md").write_text(summary + "\n")
    (reports / "shap.json").write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    return {"segments": [asdict(r) for r in results], "markdown": summary}


def _summarise(results: Sequence[SegmentAttribution]) -> str:
    fams = sorted({f for r in results for f in r.family_share})
    lines = [
        "## SHAP on the served P75 model, by segment",
        "",
        "Mean |SHAP| share of total attribution, per feature family.",
        "",
        "| segment | rows | " + " | ".join(fams) + " | top feature |",
        "|---" * (len(fams) + 3) + "|",
    ]
    for r in results:
        shares = " | ".join(f"{r.family_share.get(f, 0.0):.1%}" for f in fams)
        top = r.top_features[0][0] if r.top_features else "--"
        lines.append(f"| {r.axis}={r.bucket} | {r.rows:,} | {shares} | `{top}` |")
    return "\n".join(lines)

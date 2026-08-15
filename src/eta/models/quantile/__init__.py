from __future__ import annotations

# Must run before anything in this package imports lightgbm. See _openmp.py --
# the wrong order does not fail, it hangs.
from eta.models.quantile._openmp import preload_torch_before_lightgbm

preload_torch_before_lightgbm()

from eta.models.quantile.composition import MonotonicComposition  # noqa: E402
from eta.models.quantile.crossing import (  # noqa: E402
    CrossingReport,
    crossing_report,
    is_monotone,
    sort_rows,
)
from eta.models.quantile.lgbm import (  # noqa: E402
    LgbmQuantile,
    QuantileBundle,
    default_params,
    tune_on_validation,
)
from eta.models.quantile.nn import (  # noqa: E402
    MultiHeadQuantileNet,
    pick_device,
    pinball_loss_torch,
)

__all__ = [
    "CrossingReport",
    "LgbmQuantile",
    "MonotonicComposition",
    "MultiHeadQuantileNet",
    "QuantileBundle",
    "crossing_report",
    "default_params",
    "is_monotone",
    "pick_device",
    "pinball_loss_torch",
    "sort_rows",
    "tune_on_validation",
]

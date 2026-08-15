"""Temporal robustness: does the improvement hold across the test period, or once?

Seed standard deviation answers "how sensitive is this training procedure to random
initialisation". It does **not** answer "how confident are we that the improvement
generalises". Those get conflated constantly, and on a temporal forecasting problem
the second question is the one that matters.

Two things are measured here instead:

* **Windowed** -- split the test period into consecutive blocks and report the
  improvement in each. A single fixed test period can be unusually easy or hard; if
  the gain is +2% early and +11% late, a single headline is an average over a moving
  target and should be reported as one.

* **Paired blocks** -- per block, the *difference* in mean cost between champion and
  baseline, on the same rows. Pairing removes the block-level difficulty that
  otherwise dominates the comparison. Reported as a win count and a spread across
  blocks, deliberately **not** as a p-value: consecutive blocks of traffic are
  serially correlated, so the independence a t-test assumes does not hold, and a
  significance number here would be more precise than the data supports.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from eta.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from eta.config import CostConfig

__all__ = ["BlockResult", "WindowResult", "markdown", "paired_blocks", "temporal_windows"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WindowResult:
    window: str
    rows: int
    champion_cost: float
    baseline_cost: float
    champion_late: float
    baseline_late: float

    @property
    def improvement_pct(self) -> float:
        return 100.0 * (self.champion_cost - self.baseline_cost) / self.baseline_cost


@dataclass(frozen=True, slots=True)
class BlockResult:
    blocks: int
    champion_wins: int
    mean_diff: float
    median_diff: float
    std_diff: float
    p10_diff: float
    p90_diff: float
    worst_block_diff: float
    best_block_diff: float
    longest_losing_streak: int
    weekday_diff: float
    weekend_diff: float

    @property
    def win_rate(self) -> float:
        """Share of blocks won -- **not** a probability of superiority.

        Consecutive days are serially correlated, so these are not independent
        trials and the win rate cannot be read as a confidence level.
        """
        return self.champion_wins / self.blocks if self.blocks else 0.0


def _costs(
    actual: NDArray[np.float64], promised: NDArray[np.float64], cfg: CostConfig
) -> NDArray[np.float64]:
    from eta.models.cost import business_cost

    return business_cost(actual, promised, cfg)


def temporal_windows(
    request_datetime: pl.Series,
    actual: NDArray[np.float64],
    champion: NDArray[np.float64],
    baseline: NDArray[np.float64],
    cfg: CostConfig,
    n_windows: int = 3,
) -> list[WindowResult]:
    """Split the test period into consecutive equal-row windows and compare in each."""
    order = np.argsort(request_datetime.to_numpy())
    edges = np.linspace(0, order.size, n_windows + 1).astype(int)
    labels = ["early", "middle", "late"] if n_windows == 3 else [f"w{i}" for i in range(n_windows)]

    champ_cost = _costs(actual, champion, cfg)
    base_cost = _costs(actual, baseline, cfg)

    out: list[WindowResult] = []
    for label, lo, hi in zip(labels, edges[:-1], edges[1:], strict=True):
        idx = order[lo:hi]
        out.append(
            WindowResult(
                window=label,
                rows=int(idx.size),
                champion_cost=float(champ_cost[idx].mean()),
                baseline_cost=float(base_cost[idx].mean()),
                champion_late=float(np.mean(actual[idx] > champion[idx])),
                baseline_late=float(np.mean(actual[idx] > baseline[idx])),
            )
        )
    return out


def paired_blocks(
    request_datetime: pl.Series,
    actual: NDArray[np.float64],
    champion: NDArray[np.float64],
    baseline: NDArray[np.float64],
    cfg: CostConfig,
    block: str = "1d",
) -> BlockResult:
    """Per-day paired cost difference, champion minus baseline. Negative favours champion."""
    champ_cost = _costs(actual, champion, cfg)
    base_cost = _costs(actual, baseline, cfg)

    frame = pl.DataFrame({"t": request_datetime, "diff": champ_cost - base_cost}).with_columns(
        pl.col("t").dt.truncate(block).alias("block")
    )

    per_block = (
        frame.group_by("block")
        .agg(pl.col("diff").mean().alias("mean_diff"))
        .sort("block")
        .with_columns(pl.col("block").dt.weekday().alias("dow"))
    )
    diffs = per_block["mean_diff"].to_list()
    if not diffs:
        msg = "no blocks to compare"
        raise ValueError(msg)

    # Longest run of consecutive blocks the champion lost -- a regime that persists
    # for a week is a different problem from the same number of scattered bad days.
    streak = worst_streak = 0
    for d in diffs:
        streak = streak + 1 if d >= 0 else 0
        worst_streak = max(worst_streak, streak)

    weekday = per_block.filter(pl.col("dow") <= 5)["mean_diff"].to_list()
    weekend = per_block.filter(pl.col("dow") > 5)["mean_diff"].to_list()

    result = BlockResult(
        blocks=len(diffs),
        champion_wins=sum(1 for d in diffs if d < 0),
        mean_diff=st.fmean(diffs),
        median_diff=st.median(diffs),
        std_diff=st.stdev(diffs) if len(diffs) > 1 else 0.0,
        p10_diff=float(np.percentile(diffs, 10)),
        p90_diff=float(np.percentile(diffs, 90)),
        worst_block_diff=max(diffs),
        best_block_diff=min(diffs),
        longest_losing_streak=worst_streak,
        weekday_diff=st.fmean(weekday) if weekday else float("nan"),
        weekend_diff=st.fmean(weekend) if weekend else float("nan"),
    )
    log.info(
        "paired_blocks",
        block=block,
        blocks=result.blocks,
        champion_wins=result.champion_wins,
        mean_diff=round(result.mean_diff, 3),
        worst_block=round(result.worst_block_diff, 3),
    )
    return result


def markdown(
    windows: Sequence[WindowResult], blocks: BlockResult, champion: str, baseline: str
) -> str:
    lines = [
        f"## Temporal robustness: {champion} vs {baseline}",
        "",
        "### Consecutive windows of the test period",
        "",
        "| window | rows | champion | baseline | improvement | champion late | baseline late |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in windows:
        lines.append(
            f"| {w.window} | {w.rows:,} | {w.champion_cost:.1f} | {w.baseline_cost:.1f} | "
            f"**{w.improvement_pct:+.1f}%** | {w.champion_late:.1%} | {w.baseline_late:.1%} |"
        )

    spread = max(w.improvement_pct for w in windows) - min(w.improvement_pct for w in windows)
    lines += [
        "",
        f"Spread across windows: **{spread:.1f}pp**. A single test period can be unusually "
        "easy or hard; this is what the headline is an average over.",
        "",
        "### Paired daily blocks",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| blocks (days) | {blocks.blocks} |",
        f"| champion wins | **{blocks.champion_wins}** ({blocks.win_rate:.1%} of days) |",
        f"| mean daily difference | **{blocks.mean_diff:+.2f}** |",
        f"| median daily difference | **{blocks.median_diff:+.2f}** |",
        f"| p10 / p90 | {blocks.p10_diff:+.2f} / {blocks.p90_diff:+.2f} |",
        f"| best / worst day | {blocks.best_block_diff:+.2f} / {blocks.worst_block_diff:+.2f} |",
        f"| longest losing streak | **{blocks.longest_losing_streak} days** |",
        f"| weekday mean | {blocks.weekday_diff:+.2f} |",
        f"| weekend mean | {blocks.weekend_diff:+.2f} |",
        "",
        "Negative favours the champion. Pairing removes block-level difficulty, which "
        "otherwise dominates the comparison.",
        "",
        "**The win rate is not a probability of superiority, and no p-value is reported.** "
        "Consecutive days of traffic are serially correlated, so these are not independent "
        "trials; a significance figure would be more precise than the data supports. The "
        "losing streak and the weekday/weekend split are included because a regime that "
        "persists is a different problem from the same number of scattered bad days.",
    ]
    return "\n".join(lines)

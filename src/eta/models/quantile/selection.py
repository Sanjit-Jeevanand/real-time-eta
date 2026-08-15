"""Choosing the crossing strategy on validation, and only then touching test.

The first version of Phase 6 chose the served crossing strategy by comparing
business cost **on the test split**, then reported that same test cost as the
headline. That makes the test set a model-selection set: the reported number is the
minimum of three draws, so it is optimistically biased, and the bias is worst
exactly when the candidates are close -- which they were (477.5 / 480.0 / 480.1).

The rule this module enforces:

    selection  reads validation
    reporting  reads test
    and nothing reads test before the choice is frozen

`select_champion` therefore accepts validation costs only. It has no parameter that
could carry a test score, so the discipline is structural rather than a convention
someone has to remember.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["Selection", "select_champion"]


@dataclass(frozen=True, slots=True)
class Selection:
    champion: str
    val_cost: float
    val_cost_std: float
    ranking: list[tuple[str, float]]
    margin: float

    @property
    def margin_exceeds_seed_spread(self) -> bool:
        """Is the gap to the runner-up larger than this model's own seed spread?

        Deliberately **not** a significance test, and deliberately not used to
        override the choice. Seed variability measures sensitivity to
        initialisation; it is not a standard error for the difference between two
        strategies, and treating it as one would manufacture confidence the data
        does not contain. The selection rule is simply "lowest validation cost".
        This flag exists only so the write-up can say when that winner was a
        near-tie rather than a clear result.
        """
        return self.margin > self.val_cost_std

    def markdown(self) -> str:
        lines = [
            "| strategy | validation cost | selected |",
            "|---|---|---|",
        ]
        for name, cost in self.ranking:
            mark = " **<-- champion**" if name == self.champion else ""
            lines.append(f"| {name} | {cost:.1f} |{mark} |")
        lines += [
            "",
            f"Selection rule: **lowest mean validation cost**. Margin over the runner-up is "
            f"**{self.margin:.2f}**; the champion's own seed spread is {self.val_cost_std:.2f}.",
            "",
            (
                "The margin is larger than the seed spread, so the ordering is at least not an "
                "artefact of initialisation."
                if self.margin_exceeds_seed_spread
                else "**The margin is smaller than the seed spread.** The winner is a near-tie "
                "and should be described as one; another seed could plausibly reorder these."
            ),
            "",
            "Neither statement is a significance test. Seed variability measures sensitivity to "
            "initialisation, not the standard error of a difference between strategies.",
        ]
        return "\n".join(lines)


def select_champion(val_costs: Mapping[str, Sequence[float]]) -> Selection:
    """Pick the lowest mean validation cost. Test scores are not accepted here."""
    if not val_costs:
        msg = "no candidates to select from"
        raise ValueError(msg)

    means = sorted(((name, st.fmean(v)) for name, v in val_costs.items()), key=lambda kv: kv[1])
    champion, best = means[0]
    runner_up = means[1][1] if len(means) > 1 else best
    seed_std = st.stdev(val_costs[champion]) if len(val_costs[champion]) > 1 else 0.0
    return Selection(
        champion=champion,
        val_cost=best,
        val_cost_std=seed_std,
        ranking=means,
        margin=runner_up - best,
    )

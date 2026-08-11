from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from eta.features.context import BatchContext, OnlineContext

__all__ = [
    "REGISTRY",
    "Family",
    "Feature",
    "FeatureRegistry",
]


class Family(StrEnum):
    ROUTE = "route"
    TEMPORAL = "temporal"
    CONGESTION = "congestion"
    ZONE = "zone"
    WEATHER = "weather"


type DType = pl.DataType | type[pl.DataType]
type BatchFn = Callable[["BatchContext"], pl.Expr]
type OnlineFn = Callable[["OnlineContext"], float | int | bool | None]


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    family: Family
    dtype: DType
    batch: BatchFn
    online: OnlineFn
    redis_keys: tuple[str, ...] = ()

    def batch_expr(self, ctx: BatchContext) -> pl.Expr:
        return self.batch(ctx).cast(self.dtype).alias(self.name)

    def online_value(self, ctx: OnlineContext) -> float | int | bool | None:
        return self.online(ctx)


@dataclass(slots=True)
class FeatureRegistry:
    _features: dict[str, Feature] = field(default_factory=dict)

    def register(self, feature: Feature) -> Feature:
        if feature.name in self._features:
            msg = f"duplicate feature name: {feature.name}"
            raise ValueError(msg)
        self._features[feature.name] = feature
        return feature

    def add(
        self,
        name: str,
        family: Family,
        dtype: DType,
        batch: BatchFn,
        online: OnlineFn,
        redis_keys: tuple[str, ...] = (),
    ) -> Feature:
        return self.register(Feature(name, family, dtype, batch, online, redis_keys))

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features.values())

    def __contains__(self, name: object) -> bool:
        return name in self._features

    def __getitem__(self, name: str) -> Feature:
        return self._features[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._features)

    def by_family(self, family: Family) -> tuple[Feature, ...]:
        return tuple(f for f in self._features.values() if f.family == family)

    def counts(self) -> Mapping[Family, int]:
        return {fam: len(self.by_family(fam)) for fam in Family}

    def required_redis_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for f in self._features.values():
            keys.update(f.redis_keys)
        return tuple(sorted(keys))

    def select(self, names: Iterable[str] | None = None) -> tuple[Feature, ...]:
        if names is None:
            return tuple(self._features.values())
        return tuple(self._features[n] for n in names)

    def batch_frame(self, ctx: BatchContext, names: Sequence[str] | None = None) -> pl.LazyFrame:
        feats = self.select(names)
        return ctx.frame.select([f.batch_expr(ctx) for f in feats])

    def online_row(
        self, ctx: OnlineContext, names: Sequence[str] | None = None
    ) -> dict[str, float | int | bool | None]:
        return {f.name: f.online_value(ctx) for f in self.select(names)}


REGISTRY: Final = FeatureRegistry()

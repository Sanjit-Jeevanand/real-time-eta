from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "SEGMENT_AXES",
    "Quantile",
    "SegmentAxis",
    "TimeBucket",
    "TripLengthBucket",
    "WeatherBucket",
    "ZoneDensityBucket",
    "segment_key",
]

type Quantile = float


class SegmentAxis(StrEnum):
    TIME = "time"
    TRIP_LENGTH = "trip_length"
    ZONE_DENSITY = "zone_density"
    WEATHER = "weather"


class TimeBucket(StrEnum):
    PEAK_AM = "peak_am"
    PEAK_PM = "peak_pm"
    OFF_PEAK = "off_peak"
    LATE_NIGHT = "late_night"


class TripLengthBucket(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ZoneDensityBucket(StrEnum):
    MANHATTAN_CORE = "manhattan_core"
    OUTER_BOROUGH = "outer_borough"
    AIRPORT = "airport"


class WeatherBucket(StrEnum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"


SEGMENT_AXES: dict[SegmentAxis, type[StrEnum]] = {
    SegmentAxis.TIME: TimeBucket,
    SegmentAxis.TRIP_LENGTH: TripLengthBucket,
    SegmentAxis.ZONE_DENSITY: ZoneDensityBucket,
    SegmentAxis.WEATHER: WeatherBucket,
}


def segment_key(buckets: Sequence[StrEnum]) -> str:
    return "|".join(sorted(str(b) for b in buckets))

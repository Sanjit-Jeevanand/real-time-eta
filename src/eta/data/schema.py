from __future__ import annotations

from typing import Final

__all__ = [
    "AIRPORT_ZONE_IDS",
    "EVAL_ONLY_COLUMNS",
    "MAX_VALID_ZONE_ID",
    "NYC_TZ",
    "POST_HOC_COLUMNS",
    "REQUEST_TIME_KNOWN",
    "TLC_BASE_URL",
    "TLC_READ_COLUMNS",
    "UNKNOWN_ZONE_IDS",
    "ZONE_LOOKUP_URL",
]

NYC_TZ: Final = "America/New_York"

TLC_BASE_URL: Final = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_LOOKUP_URL: Final = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

TLC_READ_COLUMNS: Final = (
    "hvfhs_license_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
)

MAX_VALID_ZONE_ID: Final = 263
UNKNOWN_ZONE_IDS: Final = (264, 265)
AIRPORT_ZONE_IDS: Final = (1, 132, 138)


REQUEST_TIME_KNOWN: Final = frozenset(
    {
        "request_datetime",
        "hvfhs_license_num",
        "pu_zone",
        "do_zone",
        "temp_c",
        "wind_ms",
        "visibility_m",
        "precip_mm_h",
        "snow_depth_cm",
        "seg_time",
        "seg_zone_density",
        "seg_weather",
    }
)

POST_HOC_COLUMNS: Final = frozenset(
    {
        "on_scene_datetime",
        "pickup_datetime",
        "dropoff_datetime",
        "trip_miles",
        "trip_time",
        "total_time_s",
        "dispatch_approach_s",
        "curb_wait_s",
        "trip_duration_s",
    }
)

EVAL_ONLY_COLUMNS: Final = frozenset({"seg_trip_length"})

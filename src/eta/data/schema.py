from __future__ import annotations

from typing import Final

__all__ = [
    "AIRPORT_ZONE_IDS",
    "MAX_VALID_ZONE_ID",
    "NYC_TZ",
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

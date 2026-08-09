from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from eta.data.schema import NYC_TZ
from eta.data.weather import (
    STATION_COORDS,
    STATIONS,
    join_weather,
    parse_isd,
    station_by_zone,
)

CSV_HEADER = "STATION,DATE,TMP,WND,VIS,AA1,AJ1\n"


def _write(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "station.csv"
    p.write_text(CSV_HEADER + "\n".join(rows) + "\n")
    return p


def test_parses_real_field_encodings(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        [
            '725053,2023-01-01T00:06:00,"+0100,5","030,5,N,0015,5","002414,5,N,5","01,0002,3,1",',
        ],
    )
    out = parse_isd(csv, "central_park").collect()
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["temp_c"] == pytest.approx(10.0)
    assert r["wind_ms"] == pytest.approx(1.5)
    assert r["visibility_m"] == 2414
    assert r["precip_mm_h"] == pytest.approx(0.2)
    assert r["station"] == "central_park"


def test_missing_sentinels_become_null_not_extremes(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        [
            '725053,2023-01-01T00:06:00,"+9999,9","999,9,C,9999,9","999999,9,9,9","01,9999,9,9",',
        ],
    )
    r = parse_isd(csv, "central_park").collect().row(0, named=True)
    assert r["temp_c"] is None
    assert r["wind_ms"] is None
    assert r["visibility_m"] is None
    assert r["precip_mm_h"] is None


def test_negative_temperature_is_not_mistaken_for_a_sentinel(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-15T00:06:00,"-0099,5","030,5,N,0015,5","002414,5,N,5",,'],
    )
    r = parse_isd(csv, "central_park").collect().row(0, named=True)
    assert r["temp_c"] == pytest.approx(-9.9)


def test_multi_hour_accumulation_is_normalised_to_a_rate(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-01T00:06:00,"+0100,5","030,5,N,0015,5","002414,5,N,5","06,0060,3,1",'],
    )
    r = parse_isd(csv, "central_park").collect().row(0, named=True)
    assert r["precip_mm_h"] == pytest.approx(1.0)


def test_sub_hourly_reports_collapse_to_one_row_per_hour(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        [
            '725053,2023-01-01T00:06:00,"+0100,5","030,5,N,0010,5","004000,5,N,5","01,0002,3,1",',
            '725053,2023-01-01T00:31:00,"+0120,5","030,5,N,0020,5","002000,5,N,5","01,0008,3,1",',
            '725053,2023-01-01T01:06:00,"+0140,5","030,5,N,0030,5","009000,5,N,5",,',
        ],
    )
    out = parse_isd(csv, "central_park").collect().sort("hour")
    assert out.height == 2
    first = out.row(0, named=True)
    assert first["temp_c"] == pytest.approx(11.0)
    assert first["visibility_m"] == 2000
    assert first["precip_mm_h"] == pytest.approx(0.8)


def test_timestamps_convert_from_utc_to_new_york(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-01T05:06:00,"+0100,5","030,5,N,0015,5","002414,5,N,5",,'],
    )
    hour = parse_isd(csv, "central_park").collect()["hour"][0]
    assert hour.hour == 0
    assert hour.utcoffset() == dt.timedelta(hours=-5)


def test_station_ids_are_distinct() -> None:
    assert len(set(STATIONS.values())) == len(STATIONS)
    assert set(STATIONS) == set(STATION_COORDS)


def test_every_station_has_plausible_nyc_coordinates() -> None:
    for name, (lat, lon) in STATION_COORDS.items():
        assert 40.4 < lat < 41.0, name
        assert -74.4 < lon < -73.5, name


def test_nearest_station_picks_the_closest_by_distance() -> None:
    zones = pl.DataFrame(
        {
            "zone_id": pl.Series([132, 138, 1, 161], dtype=pl.UInt16),
            "centroid_lat": [40.6470, 40.7744, 40.6918, 40.7580],
            "centroid_lon": [-73.7865, -73.8736, -74.1740, -73.9855],
        }
    )
    out = station_by_zone(zones)
    assert out["station"].to_list() == ["jfk", "lga", "ewr", "central_park"]


def test_newark_zone_is_no_longer_stranded_on_central_park() -> None:
    zones = pl.DataFrame(
        {
            "zone_id": pl.Series([1], dtype=pl.UInt16),
            "centroid_lat": [40.6918],
            "centroid_lon": [-74.1740],
        }
    )
    out = station_by_zone(zones)
    assert out["station"][0] == "ewr"
    assert out["station_km"][0] < 2.0


def test_join_matches_across_time_units(tmp_path: Path) -> None:
    hour = dt.datetime(2023, 6, 1, 18, tzinfo=ZoneInfo(NYC_TZ))
    weather = pl.DataFrame(
        {
            "hour": pl.Series([hour], dtype=pl.Datetime("us", NYC_TZ)),
            "station": ["central_park"],
            "temp_c": [21.0],
            "wind_ms": [3.0],
            "visibility_m": [16000],
            "precip_mm_h": [0.0],
            "snow_depth_cm": [0],
        }
    )
    trips = pl.LazyFrame(
        {
            "request_datetime": pl.Series(
                [hour + dt.timedelta(minutes=42)], dtype=pl.Datetime("ns", NYC_TZ)
            ),
            "pu_zone": pl.Series([161], dtype=pl.UInt16),
        }
    )
    out = join_weather(trips, weather).collect()
    assert out["temp_c"][0] == pytest.approx(21.0)
    assert "hour" not in out.columns
    assert "station" not in out.columns


def test_negative_temp_sentinel_is_also_missing(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-01T00:06:00,"-9999,9","030,5,N,0015,5","002414,5,N,5",,'],
    )
    assert parse_isd(csv, "central_park").collect().row(0, named=True)["temp_c"] is None


def test_zero_hour_precip_period_does_not_divide_by_zero(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-01T00:06:00,"+0100,5","030,5,N,0015,5","002414,5,N,5","00,0010,3,1",'],
    )
    assert parse_isd(csv, "central_park").collect().row(0, named=True)["precip_mm_h"] is None


def test_implausible_precip_period_is_rejected(tmp_path: Path) -> None:
    csv = _write(
        tmp_path,
        ['725053,2023-01-01T00:06:00,"+0100,5","030,5,N,0015,5","002414,5,N,5","99,0010,3,1",'],
    )
    assert parse_isd(csv, "central_park").collect().row(0, named=True)["precip_mm_h"] is None

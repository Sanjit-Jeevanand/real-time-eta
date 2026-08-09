# Data Card

Generated 2026-08-09 by `make data-card`.

## Source

| | |
|---|---|
| Dataset | NYC TLC High Volume For-Hire Vehicle trip records |
| Operators | Uber (HV0003), Lyft (HV0005) |
| Window | 2023-01-01 to 2024-03-11 |
| Months | 15 |
| Weather | NOAA ISD hourly, Central Park / LGA / JFK |
| Licence | Public domain (NYC Open Data) |

## Volume

| | rows |
|---|---:|
| Raw | 292,793,886 |
| Removed by filters | 14,615,930 (4.9919%) |
| Kept | 278,177,956 |
| Enriched | 278,177,956 |

## Filter summary

Each rule is evaluated independently, so the counts overlap and do not sum
to the net removal. A row with a null timestamp trips four rules at once.

| rule | rejected alone | % | reason |
|---|---:|---:|---|
| `zones_routable` | 11,889,275 | 4.061 | zones 264/265 are Unknown/N-A and have no centroid to route from |
| `timestamps_ordered` | 2,858,001 | 0.976 | request <= pickup <= dropoff; anything else is a clock or join error |
| `positive_duration` | 293,466 | 0.100 | non-positive total_time cannot be a promise |
| `distance_consistent` | 50,071 | 0.017 | zero distance with non-zero trip time: the ride never moved |
| `duration_under_6h` | 1,665 | 0.001 | trips over 6h are meter-left-running artefacts, not rides |
| `dispatch_under_2h` | 1,446 | 0.001 | waiting over 2h for a car is a stuck record |
| `speed_plausible` | 694 | 0.000 | implied speed over 80 mph is impossible in NYC |
| `timestamps_present` | 190 | 0.000 | request/pickup/dropoff must all exist to form the target |

Independent rejections total 15,094,808 against 14,615,930 net removals (1.03x overlap).

## Target

`total_time_s = dropoff_datetime - request_datetime`, the user-facing promise.
Predicted from request-time information only.

| statistic | minutes |
|---|---:|
| mean | 23.5 |
| std | 13.5 |
| min | 0.3 |
| p50 | 20.2 |
| p75 | 29.3 |
| p90 | 40.8 |
| p95 | 49.5 |
| p99 | 70.1 |
| max | 359.9 |

Mean/median ratio 1.161. The distribution has a long right tail: p99 is 3.5x the median. This is why the project optimises a quantile of the predictive distribution rather than its mean.

## Splits

Wall-clock, never random. `cal` is held out from both training and test so
isotonic recalibration and CQR have data neither fitted nor evaluated on.

| split | from | to | rows | share |
|---|---|---|---:|---:|
| train | 2023-01-01 | 2023-10-01 | 163,922,693 | 58.9% |
| cal | 2023-10-01 | 2023-11-15 | 27,766,245 | 10.0% |
| val | 2023-11-15 | 2023-12-15 | 19,015,319 | 6.8% |
| test | 2023-12-15 | 2024-01-15 | 18,180,515 | 6.5% |
| holdout | 2024-01-15 | 2024-03-11 | 35,915,181 | 12.9% |
| beyond | - | - | 13,378,003 | 4.8% |

## Segments

Frozen here and not re-cut afterwards: the Phase 7 miscalibration result and
the Phase 10 cost attribution are only comparable on a fixed grid.

### `seg_time`

| bucket | rows | share | p50 (min) | p90 (min) |
|---|---:|---:|---:|---:|
| peak_am | 45,431,805 | 16.3% | 20.0 | 40.5 |
| peak_pm | 63,660,269 | 22.9% | 20.9 | 43.6 |
| off_peak | 125,456,129 | 45.1% | 20.3 | 41.4 |
| late_night | 43,629,753 | 15.7% | 19.4 | 35.9 |

### `seg_trip_length`

**Evaluation only.** Derived from `trip_miles`, the distance actually
travelled, which is not known when the ETA is quoted. Safe for slicing
results; unusable as a model feature or as a Mondrian CQR conditioning
variable until Phase 3 supplies OSRM route distance as a request-time proxy.

| bucket | rows | share | p50 (min) | p90 (min) |
|---|---:|---:|---:|---:|
| short | 100,024,559 | 36.0% | 12.8 | 20.1 |
| medium | 133,322,720 | 47.9% | 22.8 | 36.0 |
| long | 44,830,677 | 16.1% | 39.5 | 63.0 |

### `seg_zone_density`

| bucket | rows | share | p50 (min) | p90 (min) |
|---|---:|---:|---:|---:|
| manhattan_core | 111,217,351 | 40.0% | 21.0 | 40.4 |
| outer_borough | 157,776,833 | 56.7% | 19.0 | 38.2 |
| airport | 9,183,772 | 3.3% | 40.0 | 65.4 |

### `seg_weather`

| bucket | rows | share | p50 (min) | p90 (min) |
|---|---:|---:|---:|---:|
| clear | 250,748,918 | 90.1% | 20.2 | 40.7 |
| rain | 27,147,120 | 9.8% | 20.5 | 41.8 |
| snow | 281,918 | 0.1% | 21.2 | 43.5 |

## Mondrian grid viability

Segment-conditional CQR needs a calibration set per cell of the full grid.
Cells thinner than `calibration.min_segment_samples` fall back to the global map,
so the count below is what decides whether conditional coverage is reachable.

| | |
|---|---:|
| Cells in the grid | 72 |
| Calibration rows | 27,766,245 |
| Cells below 5,000 rows | 12 |
| Rows in those cells | 21,990 |
| Smallest cell | 86 |

Thinnest cells:

| seg_time | seg_trip_length | seg_zone_density | seg_weather | rows |
|---|---|---|---|---|
| late_night | short | airport | rain | 86 |
| peak_am | short | airport | rain | 118 |
| peak_pm | short | airport | rain | 241 |
| off_peak | short | airport | rain | 674 |
| late_night | medium | airport | rain | 911 |
| peak_am | medium | airport | rain | 1,257 |
| peak_am | short | airport | clear | 1,560 |
| late_night | short | airport | clear | 2,275 |


## Missingness

| column | null % | note |
|---|---:|---|
| `dispatch_approach_s` | 27.50 | absent when `on_scene_datetime` is not reported |
| `curb_wait_s` | 27.50 | absent when `on_scene_datetime` is not reported |
| `trip_duration_s` | 0.00 | derived from pickup/dropoff only, so always present |
| `temp_c` | 0.00 | NOAA gap beyond the 2h forward-fill |
| `wind_ms` | 0.23 | NOAA gap beyond the 2h forward-fill |
| `visibility_m` | 0.06 | NOAA gap beyond the 2h forward-fill |
| `precip_mm_h` | 0.00 | NOAA gap beyond the 2h forward-fill |
| `snow_depth_cm` | 0.01 | NOAA gap beyond the 2h forward-fill |

`on_scene_datetime` is reported by Uber and effectively absent for Lyft, so the component decomposition covers only part of the data. The target itself is complete. Components are left null rather than zero-filled: a zero would read as an instant pickup and bias the decomposition toward the operator that reports the field.

## Leakage boundary

Known at request time (12 columns):

`do_zone`, `hvfhs_license_num`, `precip_mm_h`, `pu_zone`, `request_datetime`, `seg_time`, `seg_weather`, `seg_zone_density`, `snow_depth_cm`, `temp_c`, `visibility_m`, `wind_ms`

Post-hoc, never a feature (9 columns):

`curb_wait_s`, `dispatch_approach_s`, `dropoff_datetime`, `on_scene_datetime`, `pickup_datetime`, `total_time_s`, `trip_duration_s`, `trip_miles`, `trip_time`

Enforced by `tests/test_leakage.py`, which runs in CI.

## Known limitations

- Unroutable zones (264 Unknown, 265 N/A) account for 4.06% of raw rows and are dropped. These are legitimate trips whose endpoint the feed never resolved, not corrupt records. Coverage limitation, not a cleaning statistic.
- Weather station assignment is a placeholder: airport zones take their own station, everything else takes Central Park. Phase 3 replaces it with true nearest-centroid once zone geometry exists.
- Zone-level geography only. TLC stopped publishing coordinates in 2016, so there is no sub-zone spatial resolution.
- Quoted ETAs affect cancellation, and cancelled trips never enter this feed. The dataset is censored by the very predictions it is used to train.

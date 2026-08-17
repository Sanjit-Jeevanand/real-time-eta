# real-time-eta

Real-time delivery ETA prediction on NYC TLC high-volume for-hire vehicle data.
Predicts `dropoff_datetime − request_datetime` — the user-facing promise, not just
travel time — as a set of quantiles with a distribution-free coverage guarantee.

278,177,956 trips over 15 months. Quantile LightGBM, conformalised. Sub-25ms serving
target on 2 vCPU.

---

## The serving quantile is derived, not tuned

This is the load-bearing idea in the project, so it goes first.

Being late is worse than being early. A rider told "8 minutes" who waits 14 is angry;
one told "14 minutes" who waits 8 is mildly pleased. Any honest objective has to be
asymmetric:

```
cost(actual, promised) = λ_late  · max(0, actual − promised)
                       + λ_early · max(0, promised − actual)
```

### Deriving q*

Let `F` be the predictive distribution of `actual`. Expected cost of promising `p`:

```
E[cost(p)] = λ_late · E[(actual − p)⁺] + λ_early · E[(p − actual)⁺]
```

Differentiating with respect to `p`:

```
d/dp E[cost(p)] = −λ_late · P(actual > p) + λ_early · P(actual ≤ p)
                = −λ_late · (1 − F(p))    + λ_early · F(p)
```

Setting to zero:

```
λ_early · F(p) = λ_late · (1 − F(p))
F(p) · (λ_early + λ_late) = λ_late
```

```
                        λ_late
        F(p*)  =  ─────────────────  =  q*
                  λ_late + λ_early
```

**The cost-minimising promise is the q* quantile of the predictive distribution.**
The second derivative is `(λ_late + λ_early)·f(p) ≥ 0`, so it is a minimum.

### It is exactly the pinball loss

Pinball loss at level `q` is `ρ_q(a, p) = q·(a − p)⁺ + (1 − q)·(p − a)⁺`.
Substituting `q = λ_late/(λ_late + λ_early)`, so `1 − q = λ_early/(λ_late + λ_early)`:

```
ρ_q*(a, p) = [λ_late · (a − p)⁺ + λ_early · (p − a)⁺] / (λ_late + λ_early)
           = cost(a, p) / (λ_late + λ_early)
```

Business cost and pinball loss at `q*` are **the same objective up to a positive
constant**. Minimising one minimises the other. That is why the models are trained with
`objective='quantile'` at `q*` and not at a level chosen by a sweep.

Both facts are asserted numerically in `tests/test_config.py`, not just written down here.

### What the ratio implies

| late : early | q* | serve |
|---|---|---|
| 2 : 1 | 0.667 | P67 |
| **3 : 1** | **0.750** | **P75** |
| 5 : 1 | 0.833 | P83 |

The project assumes **3:1**, stated as an assumption in the honesty ledger below, with a
sensitivity sweep. Serving P75 is not a hyperparameter that happened to win — it falls
out of that ratio.

### It cannot drift

`optimal_quantile` is a computed property on a frozen `CostConfig` with no setter, and a
root validator rejects any configuration whose implied `q*` is not among the trained
quantiles:

```
cost ratio 5.0:1.0 implies q*=0.8333, which is not in
model.quantiles=(0.05, 0.5, 0.75, 0.9, 0.95). Either train that quantile or
change the cost ratio -- do not serve the nearest available level.
```

Change the ratio and either the served quantile moves with it or the config fails to
load. There is no third option.

---

## Why not MAE

MAE and RMSE are reported throughout as secondary metrics, deliberately, because they
**disagree** with business cost — and the disagreement is the point.

An L2-optimised model predicts the conditional mean. Under a 3:1 asymmetry the mean is
systematically too optimistic, and the measured result says so:

| model | business cost | MAE (min) | **late rate** | vs 25% target |
|---|---|---|---|---|
| LightGBM, L2 loss | 515.1 ± 0.2 | 4.61 | **39.0%** | **+14.0pp** |
| historical mean by zone-pair × hour | 716.4 | 5.76 | 42.1% | +17.1pp |
| OSRM free-flow × learned multiplier | 936.3 | 6.68 | 65.0% | +40.0pp |

*Three seeds, mean ± std. Fitted on the training split of a 6.87M-row feature matrix; evaluated on the 546,261-row test split. Every row carries a digest of the evaluated population, and the leaderboard refuses to render if two models were scored on different ones.*

The **economic target** late rate is `1 − q* = 25%` — the rate that minimises expected
cost under the assumed 3:1 ratio, not a rate the model is guaranteed to realise. The best L2 model is late **39.0%** of the time —
fourteen percentage points too often — because it is answering a different question. It
minimises squared error faithfully; squared error just is not what the business pays.

That gap is what the quantile models in Phase 6 exist to close, and it is why the
headline comparison is against this L2 model rather than against a weak baseline.

### Closing it

| model | business cost | vs L2 | MAE (min) | **late rate** | vs 25% target |
|---|---|---|---|---|---|
| **LightGBM P75, composed** (champion) | **476.0 ± 1.3** | **−7.6%** | 5.62 | **22.7%** | −2.3pp |
| LightGBM P75, sorted | 479.7 ± 1.1 | −6.9% | 5.71 | 22.5% | −2.5pp |
| LightGBM P75, raw | 479.9 ± 1.3 | −6.9% | 5.71 | 22.5% | −2.5pp |
| LightGBM, L2 loss | 515.1 ± 0.2 | — | 4.61 | 39.0% | +14.0pp |
| multi-head NN, ordered *(untuned)* | 603.7 ± 12.2 | +17.2% | 8.80 | 11.0% | −14.0pp |
| multi-head NN, free *(untuned)* | 658.4 ± 43.2 | +27.8% | 9.90 | 8.5% | −16.5pp |

Same 546,261-row test split, same population digest, three seeds. The late rate moves
from **39.0% to 22.7%** against a 25% economic target, and business cost falls 7.6% —
while MAE gets *worse* (4.61 → 5.62). That is the whole thesis in one table: the quantile
model is not a better predictor of the average, it is a better answer to the question
being asked.

**The champion was chosen on validation, not on test.** The three crossing strategies were
compared on the validation split (composed 516.8, sorted 519.7, raw 519.8), the winner was
frozen, and the test split was then read once. An earlier version of this table selected on
test and is kept, labelled, as `reports/quantile_TEST_SELECTED_CONTAMINATED.md`.

### The headline caveat: this gain decays across the test period

| window | champion | L2 | improvement | champion late | L2 late |
|---|---|---|---|---|---|
| early | 511.7 | 592.7 | **−13.7%** | 27.4% | 44.5% |
| middle | 477.1 | 507.5 | **−6.0%** | 22.2% | 38.2% |
| late | 435.2 | 447.5 | **−2.7%** | 18.9% | 34.9% |

**Spread 10.9pp.** The −7.6% is an average over an advantage that falls fivefold from the
start of the test period to the end. Quote it as *"−7.6% averaged over the test period,
ranging −13.7% to −2.7% across consecutive thirds"* — not as a stable rate. Phase 7
independently found temporal shift in the same data (conformal coverage at 89.1% against a
90% target); two measurements pointing at the same drift is worth more than either alone.

Paired daily blocks: the champion wins **28 of 31 days**, longest losing streak **one day**,
median daily difference −24.3. So it is a trend, not a few bad days. The win rate is not a
probability of superiority and no p-value is reported — consecutive days are serially
correlated, so they are not independent trials.

**The plan projected 20–30%. The measured figure is −7.6%, and the plan's bullet has been
rewritten to say so.** The L2 baseline was fitted on identical features with a comparable
budget over three seeds, so it is a strong comparator rather than a strawman, and closing
a 16pp late-rate gap is where the value landed. A projection that survives contact with
the data becomes a result; one that does not becomes a corrected claim.

The neural comparator loses, clearly and reproducibly. It is reported because it was run,
not because it helps.

### Two things that were expected and did not happen

**Modelling the components separately and summing them loses**, in all 12 segments —
including the peak buckets where the plan expected it to win (direct 477.2 vs decomposed
486.6). The reason is not a modelling failure but an identity that does not hold:
`total = dispatch + curb_wait + trip_duration` is exact per row, yet
`Q(A) + Q(B) + Q(C) ≥ Q(A+B+C)`, with equality only under perfect comonotonicity. Summing
three P75s overshoots the P75 of the total by **0.54 min on 71.5% of rows**, dragging the
late rate down to 18.1% — conservatism that costs money under an asymmetric objective.

**Congestion does not dominate at PM peak.** Exact TreeSHAP sliced by segment puts route
features at 46–62% of attribution *everywhere*, PM peak included (56.7%, identical to
off-peak). Congestion's largest share is medium trips at 19.8%, not any time bucket.
Weather behaves as it should: 1.6% on clear rows, 6.4% on rain.

---

## Calibration: the aggregate is not the number

Globally the P75 model covers 77.6% against a nominal 75% — a 2.6pp gap that would let
anyone call it calibrated. Per segment the spread is **10.6 percentage points**: short
trips over-covered by +8.9pp, medium trips under-covered by −1.7pp. Both directions are
wrong and they cancel in the average, which is the entire reason the average is not
reported alone.

| approach | worst-segment gap (interval) | interval coverage | mean width |
|---|---|---|---|
| raw | 3.4 ± 0.1pp | 88.19% | 18.1 min |
| per-segment isotonic (trip length) | 2.6 ± 0.3pp | 89.11% | 18.3 min |
| split CQR | 2.7 ± 0.2pp | 89.10% | 18.4 min |
| **Mondrian CQR** | **1.6 ± 0.1pp** | 89.14% | 18.5 min |

Width sits next to coverage everywhere, because coverage alone is trivially achievable
by widening. Mondrian halves the worst-segment gap for 0.4 minutes on an 18-minute
interval.

**These intervals are not tighter than the raw quantile band — they are 2.0–2.5% wider.**
The plan projected "15–25% tighter via CQR"; that is wrong in direction, not merely in
magnitude, and the bullet has been corrected. The reason is straightforward: the raw band
*under-covers* at 88.19%, so anything that reaches the target has to widen it. A tightness
claim would need a baseline that already achieves 90% coverage, and no such baseline was
built.

### The guarantee lands at 89.1%, not 90%

CQR's coverage guarantee is finite-sample and distribution-free **given exchangeability
between calibration and test**. This project's splits are temporal by construction —
calibration precedes test, because that is the only honest way to evaluate a forecaster
— so exchangeability is violated on purpose. The measured shortfall is about 0.9pp.

The same code attains 80/90/95% exactly on exchangeable synthetic data, including when
handed a deliberately terrible interval, so this is not an implementation error. The
honest claim is: *distribution-free under exchangeability, measured at 89.1% under
temporal shift.*

---

## Coverage population

Every coverage claim here, including the conformal guarantee, is measured over one
population: **trips whose pickup and dropoff zones both have geometry** — neither is
zone 264 (Unknown) nor 265 (N/A). That excludes 4.06% of raw trips. A stated "90%
coverage" means 90% of the routable population, not 90% of all hailed trips.

---

## Layout

| package | phase | responsibility |
|---|---|---|
| `config`, `types` | 1 | typed settings; derives the serving quantile |
| `data` | 2 | TLC ingest, weather join, temporal splits, segments |
| `routing` | 3 | OSRM client, offline zone-pair matrix, zone features |
| `features` | 4 | dual-compile registry, congestion state, replay, parity |
| `models` | 5–6 | cost framework, baselines, quantile models |
| `calibration` | 7 | isotonic, CQR, Mondrian hierarchical fallback |
| `serving` | 8 | FastAPI, Treelite runtime, degradation path |
| `ops` | 9–10 | drift detection, recalibration triggers |

```bash
make setup          # deps + pre-commit hooks
make ci             # lint -> leakage -> parity -> unit
make data           # TLC ingest, weather, splits, segments, data card
make route          # OSRM matrix, zone features, detour ratios, zone reference table
make model-matrix   # feature matrix
make baselines      # three baselines x three seeds
make quantile       # quantile models, crossing rate, three fixes
make calibrate      # coverage, isotonic, CQR, Mondrian CQR
make ablation       # decomposed vs direct
make explain        # TreeSHAP by segment
make compile        # Treelite, with an equivalence check
make latency-full   # per-stage latency, real Redis + compiled models
```

---

## Honesty ledger

| Claim | What it actually is |
|---|---|
| "Real-time congestion state" | A **replayed** event stream updating a key-value store, not live Kafka/Flink. Functionally equivalent here; stated rather than implied. The replay assumes a globally ordered stream and raises on any out-of-order event — correct for a replay, where order holds by construction. The production assumption this stands in for: **Kafka partitioned by pickup zone, ordered within partition** — every congestion window is per-zone, so per-zone ordering is all the state machine needs. A late-arriving event beyond the partition's ordering guarantee would need a watermark and a correction path, neither of which exists here. |
| Business cost ratio | **Assumed** 3:1 late:early with a sensitivity sweep — not measured from a real business. |
| "Shadow traffic" | **Replayed** historical requests against the live service. |
| Drift over 8 weeks | **Replay** over 8 weeks of historical data, not wall-clock. |
| Daily prediction capacity | **Benchmarked** capacity extrapolated from sustained QPS — not observed production volume. |
| Feedback loops | **Acknowledged and bounded, not corrected.** Quoted ETAs affect cancellation, cancelled trips never enter the feed, so the training set is censored by its own predictions. Resolving it needs a randomised-quote experiment that cannot be run offline. |
| Intra-zone distance fallback | Observed medians for 245 of 263 zones; the other 18 use an area-based relationship that is an **extrapolation**, not cross-validated on those zones. |
| Conformal coverage | **Distribution-free under exchangeability, measured at 89.1% against a 90% target.** The splits are temporal, so exchangeability does not hold and the theorem does not fully apply. The gap is stated rather than rounded away. |
| "Tighter intervals" | **Not demonstrated, and measured in the opposite direction** — conformalised intervals are 2.0–2.5% *wider* than the raw band, because the raw band under-covers. Bullet 4 originally projected 15–25% tighter; it has been rewritten to the measurement. |
| Worst-segment miscalibration | **3.4 → 1.6pp** on the 90% interval, and **9.1 → 7.0pp** on the served P75. The plan projected a 12–18pp starting point; the actual starting point was far smaller, so the dramatic reduction it implies never existed to be made. |
| "Sub-25ms serving" | **Measured at p99 0.627ms on an M4** with a real Redis over TCP and Treelite-compiled models — but that is loopback, 2,000 sequential requests, no concurrency, and not the 2-vCPU target box. The budget is not discharged by it. |
| Deployment | **Nothing is deployed.** No Hetzner box, no HTTPS endpoint, no Grafana dashboard. The service runs and is tested locally; the deployment half of Phase 8 is outstanding. |
| Business cost improvement | **−7.6%** against a strong L2 baseline on identical features, and **not stable** — it ranges −13.7% to −2.7% across consecutive thirds of the test period. The plan originally projected 20–30%; that bullet has been rewritten to the measured number rather than left as a target. |
| Zone "reference speed" | An **empirical 3–5am median speed**, used as a low-congestion reference. It is deliberately *not* called a free-flow speed: 3–5am is the quietest window in the data, but nothing here shows it is uncongested in the traffic-engineering sense. It replaced an OSRM-routed reference that measured 0.61 against real 3–5am trips — and 0.87 in airport zones, whose highway-dominated trips resemble the routes that reference was built from. 251 of 263 zones use the observed value; 12 fall below the 200-trip floor and keep the routed one. The feature it feeds, `zone_speed_ratio_*`, is therefore a **relative degradation** measure, not an absolute speed. |

Every number is real. The setup is stated plainly so no one has to guess which parts are
production and which are replay.

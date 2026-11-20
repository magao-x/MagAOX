# 8 kHz accelerometer-assisted DDSPC: formulation and implementation plan

2026-11-20. Plan for review; no code has been changed. Workstation `ASTR-LT-428.local`,
branch `parkertjohnson1/accel_ddspc` at `1ff631bb`.

## Request

At `1ff631bb`, assisted mode solves once per WFS frame (2 kHz) and publishes the
four planned slots at 8 kHz. Instead, assisted mode must:

- recalculate a distinct prediction every 125 µs using the newest accelerometer sample;
- keep its command history from scheduler-acknowledged copies, including skipped slots;
- learn the accelerometer contribution by RLS alongside the WFS and DM histories;
- never treat a repeated WFS frame as a new observation.

Standard DDSPC stays as it is.

Fixed constraints:

- Keep upstream ACC/WFS synchronization; the controller does not synchronize anything itself.
- Keep the `dmtweeter-synco` output path and the current channel assignments.
- Stay causal.
- Fit the work, learning included, within the 125 µs cadence.

## 1. Notation and current model

| Symbol | Meaning | RTC value |
| --- | --- | --- |
| `m` | controlled modes | 2 |
| `T` | WFS period | 500 µs |
| `q` | DM slots per WFS frame | 4 (slot period `τ = T/q` = 125 µs) |
| `N` | WFS history, frames | 15 |
| `F` | prediction horizon, frames | 3 |
| `L` | base accelerometer window, samples | `4·accel_history` = 20 |
| `y_i` | WFS modal measurement delivered at frame `i` | `m` values |
| `u_{i,j}` | applied command of slot `j` = 0..3 of frame `i` | `m` values |
| `u_i` | lift `[u_{i,3}; u_{i,2}; u_{i,1}; u_{i,0}]`, newest slot first | `qm` values |
| `a_n` | X/Y acceleration sample, 8 kHz counter `n` | 2 values |
| `c_i` | anchor: newest ACC counter at the frame-`i` WFS decision (rule in §5) | |

`y_i` is the exposure-integrated residual. With a fixed DM/WFS latency it is linear in:

- the past disturbance;
- the individual slot commands that act during its exposure.

Lifting (Bamieh–Pearson; Chen & Francis, multirate sampled-data systems) turns this
periodic 8 kHz/2 kHz system into a time-invariant system at the frame rate, with
input `u_i` and output `y_i`.

The DDSPC predictor of Haffert et al. 2021 (JATIS 7(2), 029001; MagAO-X loPredCtrl)
applied to the lifted system is:

```text
y_f(i) = [y_{i+F}; …; y_{i+1}]            (F·m genuine future WFS frames)
u_f(i) = [u_{i+F-1}; …; u_i]              (F·q·m future slot commands)
y_f(i) = A y_p(i) + B u_p(i) + D a_p(i) + C u_f(i) + e(i)
```

- `y_p(i) = [y_i … y_{i-N+1}]`
- `u_p(i) = [u_{i-1} … u_{i-N+1}]`
- `a_p(i) = [a_{c_i} … a_{c_i-L+1}]`

The control law minimizes `‖ŷ_f‖² + λ‖u_f‖²` and applies the first block.

- **Standard (loPredCtrl):** `q = 1`, held command, `D` absent.
- **Assisted at `1ff631bb`:** `q = 4`, `D` present, solved once at the frame origin.

## 2. Proposed formulation: phase-conditioned lifted DDSPC

### 2.1 Information available at each 125 µs tick

At tick `(i, k)` the newest WFS frame is `i` and the slot to be committed is `u_{i,p}`.

**Slot phase** `p = ordinal − base_i − 1`. Here `base_i` is the newest copied ordinal at
WFS arrival, using the existing mapping. Slots `0 … p−1` of frame `i` are known exactly
from scheduler acknowledgements. A skipped slot holds the previous copy, as synco does.

**Information phase** `k = min(newest ACC counter − c_i, 3)` is the number of
accelerometer samples received since the frame origin. Normally `k = p`. If the ACC
sample is late, `k < p`.

The information set is:

```text
I(i,k,p) = { y_≤i ,  u ≤ slot (i,p−1) ,  a_≤(c_i+k) }
```

No WFS sample is interpolated, repeated or invented. Between WFS frames the newest real
frame `y_i` stays in `y_p`.

### 2.2 The four phase predictors

The target is the same genuine future WFS vector `y_f(i)` for every phase. The phase
model conditions on `k` extra samples:

```text
y_f(i) = Θ_k φ_k(i) + e_k(i)
φ_k(i) = [ u_f(i); y_p(i); u_p(i); a_{c_i} … a_{c_i−L+1}; a_{c_i+1}; …; a_{c_i+k} ]
n_k    = F·q·m + N·m + (N−1)·q·m + 2(L+k) = 24 + 30 + 112 + 40 + 2k = 206 + 2k
```

The four regressor sets are nested: `φ_k` is the leading `n_k` entries of `φ_3`.

- `Θ_0` is the existing assisted model.
- `Θ_k` for `k ≥ 1` is the linear least-squares predictor of the same frame-integrated
  WFS future, given `k` more real accelerometer samples. These samples were taken during
  the exposure of `y_{i+1}`, which is why they carry information about it.
- The prefix slots `u_{i,0..p−1}` are not extra regressors. They are already entries of
  `u_f(i)`. Only the control solve splits `u_f` into known and decision parts.

Each `Θ_k` therefore applies to any prefix length `p`, and one model per `k` is enough.

### 2.3 Multirate RLS: one information factor at 2 kHz, four exact solutions

The design maintains **one** QR-RLS over the superset regressor `φ_3` (212 columns,
6 outputs). It adds one example per genuine WFS target. The origin-`i` example matures
when `y_{i+F}` arrives, so updates occur at 2 kHz, with the existing per-update
forgetting `γ` and prior `ρI`. The existing `QRDRecursiveLeastSquares` maintains

```text
RᵀR = γ^t ρ I + Σ_j γ^(t−j) φ_3(j) φ_3(j)ᵀ ,   W = argmin of the weighted LS + prior.
```

**Lemma (nested extraction).** Let `S` be the leading `n_k` columns of `φ_3`, `S'` the
rest, and `R_SS` the leading block of `R`. A triangular factor satisfies

```text
(RᵀR)_SS = R_SSᵀ R_SS
```

This is exactly the information factor of a separate RLS on `φ_k` with the same data,
forgetting and prior `ρI_S`. Write `Z = R Wᵀ`. A separate RLS would satisfy
`R_SS Θ_kᵀ = Z_S`, so

```text
Θ_kᵀ = W_Sᵀ + R_SS⁻¹ ( R_SS' W_S'ᵀ )
```

Computing it takes one back-substitution with 6 right-hand sides. The four `Θ_k` are the
same coefficients that four independent RLS learners would produce, but come from one
update per WFS frame.

**Why not run RLS on 8 kHz examples with a repeated target?** Feeding four examples per
frame with the same `y_f(i)` would:

- weight every WFS frame four times, with strongly correlated errors;
- shorten the forgetting memory by a factor of four;
- force one coefficient set to average over phases, with unseen samples zero-filled.

That is not a least-squares estimate of any phase predictor. The design above gives each
phase predictor exactly one independent observation per WFS frame.

The workload is still multirate at 8 kHz:

- one information update per frame;
- four `Θ_k` per frame, one per phase (`Θ_3 = W`; the other three are extracted);
- 16 gain sets per frame;
- 8 kHz predictions.

### 2.4 Control at each tick

`u_f(i)` is ordered newest first, so the `p` known prefix slots are its last `p·m`
entries. Split `C_k = [C_k,v | C_k,pre]` and `Θ_k = [C_k | M_k]`. Here `M_k` holds the
known-state columns and `x_k(i) = [y_p; u_p; a …]` is the known state. The tick solves the
same Haffert cost over the remaining slots `v`:

```text
v*   = −(C_k,vᵀ C_k,v + λ_k I)⁻¹ C_k,vᵀ ( M_k x_k + C_k,pre u_pre )  ≡  K_{k,p} [x_k; u_pre]
u_{i,p} = last m rows of v*  (+ exploration while learning)
```

`λ_k = λ · max(C_kᵀ C_k)` keeps the existing scaled-penalty rule. `K_{k,p}` is precomputed
for every pair 0 ≤ k, p ≤ 3, using the same `reference_controller()` call as
`ShadowController::snapshot()`.

At each tick the DM worker only:

1. assembles about 200 values;
2. multiplies them by a `(24−2p)` × `(n_k−24+2p)` matrix;
3. renders the slot, plus exploration while learning.

The rest of `v*` is kept as the fallback plan (§5.3).

### 2.5 Consistency with the original DDSPC

1. **Same objective.** Every tick minimizes Haffert's cost on genuine future WFS frames
   with the same `λ` rule and horizon end `i+F`. The decision vector shrinks from 12 to
   9 slots across a frame. A slot-receding horizon would require targets that do not
   exist between frames.
2. **Reduction to the original method.**
   - `q = 1`, no accelerometer: loPredCtrl (unchanged standard mode).
   - `k = p = 0`: the lifted DDSPC at `1ff631bb`.
   - `L = 0`: lifted WFS-only DDSPC.
3. **No-new-information invariance.** Suppose the accelerometer adds nothing, so
   `Θ_k = Θ_0` and the prefix equals the previously planned values. Re-solving a convex
   quadratic with part of its optimum fixed returns the rest of that optimum. The 8 kHz
   recalculation then reproduces the 2 kHz plan exactly.
   - Any change between ticks comes only from new accelerometer information, an
     exploration probe, or a skipped slot.
   - The prefix enters `C_k,pre u_pre` with the probe included, so later slots correct
     for it.
   - This is a unit test (§7, T5).
4. **Learned accelerometer weighting.**
   - `D` and the phase blocks are RLS coefficients. No weight is assigned by hand.
   - If the accelerometers are uninformative, those columns converge to zero, apart from
     prior shrinkage and estimation noise. `Θ_k` then tends to `Θ_0`.
5. **Exposure integration and latency** need no explicit model.
   - `y_{i+1}` integrates the residual over its exposure. Each slot of `u_i` has its own
     column in `C`, so each slot's exposure weight and delay is learned.
   - A slot that acts only after the exposure ends has a near-zero coefficient for
     `y_{i+1}`.
   - The extra accelerometer samples are point samples taken during that exposure, and
     the regression learns how they relate to the integral.
6. **Limitation, stated plainly.** The cost penalizes only frame-integrated WFS
   residuals. Ripple inside a frame is unobservable to the WFS and so is not penalized.
   Slot-to-slot smoothness is controlled only by `λ`. A slot-difference penalty can be
   added later if PSDs show high-frequency ripple.
7. **Closed-loop identification.**
   - Future slots in `u_f(i)` depend on samples that a lower-phase model does not see.
     Original DDSPC has the same issue, because `u_{i+1}` depends on `y_{i+1}`.
   - Mitigation is the same as loPredCtrl: independent per-slot exploration while
     learning, with the fit frozen for evaluation.
   - The simulation tests measure the coefficient bias against an open-loop
     identification.

### 2.6 Lengths, horizons and work (RTC dimensions)

| Item | Value |
| --- | --- |
| WFS history `N` | 15 frames (7.5 ms) |
| Command history | 14 lifts = 56 slots |
| ACC window | `L+k` = 20–23 samples (2.5–2.9 ms); `accel_history` stays in WFS frames |
| Horizon | `F` = 3 frames; 12 − p decision slots |
| Superset model | 212 regressors × 6 outputs |
| Phase models | 206, 208, 210, 212 regressors |
| Gains | 16 `K_{k,p}` |

Per WFS frame, on the WFS thread (spare28), after the origin is published:

- the standard learning, unchanged;
- one 212-column QR update;
- three extractions (`Θ_3 = W`);
- 16 gain solves.

The assisted learner at `1ff631bb` (206 columns) is no longer trained, which roughly
offsets the cost. On the Mac the current learning decision takes 137 µs per frame. The
expected total is about 200–260 µs per 500 µs frame, to be measured. Per tick, the DM
worker (spare29) does only about 5 k multiply–adds, a few µs.

## 3. Standard DDSPC

The code path is unchanged:

- `ShadowController` standard model, held command, 2 kHz;
- RLS before the decision, as at `1ff631bb`;
- the slot plan replicated to the four slots.

`PredictiveController`, `ShadowController`, `qrd_rls` and the standard branch of
`loPredCtrlAcc_operational::step()` do not change. One behaviour also stays: while
assisted is selected, the standard fit still learns from the frame mean of the four copies.

Verification is a golden-trace test (§7, T1). Its fixture is recorded at `1ff631bb`
before any edit, and the test requires bitwise-equal commands.

## 4. Architecture

### 4.1 New learner

New files `multirate_ddspc.hpp/.cpp`, a library class in the style of the
`shadow_controller` files.

`MultirateAssistedLearner` owns:

- the y-lift and u-lift rings, keyed by WFS counter (a gap resets them, as at present);
- one `QRDRecursiveLeastSquares` over `φ_3`, reading its public `R_` and
  `prediction_matrix`;
- `extract(k)`;
- `gains()`, producing `K_{k,p}`;
- readiness: updates ≥ `n_3` (the existing `m_minUpdates` rule), with finite coefficients
  and gains;
- phase-RMSE scoring against genuine targets.

The original WFS counter is retained for each ACC anchor. Training uses the ACC samples
`c_i−L+1 … c_i+3`, which must have consecutive counters in one segment. Otherwise that
example is skipped; only the y/u histories trigger a reset.

### 4.2 Threads

- **WFS monitor thread (spare28, prio 2):** on frame `i`:
  1. Record `c_i`.
  2. Publish the origin snapshot `{i, base_i, y_p, u_p, c_i, gain-set index}`. This
     happens **before** any learning, so the first slot of the frame already uses `y_i`.
  3. Run the standard path, unchanged.
  4. Run the multirate update, extraction and gains into the inactive buffer of a
     preallocated double buffer.
  5. Swap the index under the existing command mutex.

  No allocation occurs on either thread.
- **DM worker (spare29, prio 20, existing):** `outputCandidate` for an assisted slot:
  1. Read the newest ACC samples. The worker has its own `AdcReader` cursor, or uses the
     shared ring if the reader is not multi-consumer; checked in step P1.
  2. Compute `k` and `p`.
  3. Take the prefix from `m_appliedSlots`. This needs no lock: `copied()` runs on the
     same thread, and the one-outstanding-commit rule guarantees that slot `p−1` has been
     acknowledged or skipped.
  4. Apply `K_{k,p}`, add the per-slot probe and render.

  Standard mode keeps rendering `m_slotPlan` as it does now.
- **Ring size:** `c_ring` stays at 16. `u_p` comes from the origin snapshot, not the slot
  ring.

### 4.3 Fallbacks (no new safety gates)

- **`p ≥ 4`** (WFS frame late): use the next entry of the last tick's `v*`, which already
  covers the next lifts. This uses the most recent available WFS information.
- **`k > 3`** or a negative `k`: clamp, and count `acc_phase_anomalies`.
- **ACC history invalid:** behaves as at present. Assisted is not ready, and the existing
  qualification fault removes the output.
- **Gain set not ready:** same as an unready model now.

### 4.4 INDI and configuration

`ddspc_performance` adds the following, each with a `_rate` computed over the
existing update interval:

- `fast_predictions`
- `multirate_updates`
- `extractions`
- `fast_fallbacks`
- `acc_phase_anomalies`
- `phase_hist0..3` (the `k−p` histogram)
- `fast_us_mean` / `fast_us_p99` / `fast_us_max`
- `learn_us_mean` / `learn_us_p99` / `learn_us_max`
- `learn_overruns` (learning over 500 µs)
- `rmse_phase0..3`

Kept unchanged:

- `transport_published` / `transport_skipped` / `slot_shifts`
- `matched_standard_rmse`
- `matched_assisted_rmse`, now fed from phase 0

There are no new configuration keys. One optional key is proposed in decision D4. The
`ddspc_mode`, `learn`, `predict`, `exploration_sequence` and `reset_model` properties
behave as they do now.

## 5. Decisions for review

- **D1. ACC phase anchor.** Proposed: `c_i` is the newest ACC counter at WFS arrival,
  which is the current rule and depends on the upstream 4:1 lock. A publisher-supplied
  WFS↔ACC counter association would be exact, but it requires changing
  accelSim/ad7689Ctrl, which is out of scope. `acc_phase_anomalies` and the `k−p`
  histogram expose a misaligned lock.
- **D2. Assisted learner at `1ff631bb`.** Proposed: stop training and using the
  `ShadowController` assisted model in the operational app, and replace it with
  `MultirateAssistedLearner`. Its phase 0 is the same predictor.
- **D3. Learning placement.** Proposed: one burst on the WFS thread after the origin is
  published, which needs no new thread. If the RTC p99 exceeds about 400 µs per frame,
  spread one extraction per tick across the DM worker's idle time instead.
- **D4. Render timing.** At present a slot renders as soon as it is eligible (writeLeadUs
  = 125), so the ACC sample used can be up to about 125 µs older than necessary.
  Proposed: measure on the RTC first the time from request to cutoff and the age of the
  ACC sample at render. If useful, add an app-local `dm.renderLeadUs` that waits in
  `outputCandidate` until `cutoff − renderLeadUs`, without changing `syncoOutput.hpp`.

## 6. Implementation steps

### P0 — Before any edit

1. Record the standard golden trace at `1ff631bb`: the deterministic rtc_controller plant
   with standard learning, then frozen.
2. Add the RTC-dimension benchmark harness.

### P1 — Learner (`multirate_ddspc.*`)

Implement the learner from §4.1: histories, superset QR-RLS, extraction, gains, scoring
and readiness. Add `Makefile` and tests Makefile entries.

### P2 — Operational app (`loPredCtrlAcc_operational.hpp`)

1. Publish the origin snapshot before learning.
2. Add the assisted fast path in `outputCandidate`.
3. Add per-slot exploration at the tick.
4. Add the counters and latency statistics, and the INDI fields.
5. Leave the standard branch untouched.

### P3 — Tests (§7) and the Python PSD/RMSE report

The report reuses the `signal.welch` pattern in `tests/vibration_report.py`.

### P4 — Documentation

Update `RTC_TESTING.md` (the 8 kHz checks and expected rates), `OPERATIONAL_DDSPC.md`,
and this plan's results section. Run clang-format. Make separate commits:

1. functional change and plan;
2. documentation;
3. formatting.

All of these are held until review.

## 7. Tests and validation

Unit and simulation tests use Catch2 in `tests/`. The simulation's true disturbance is
used only for scoring, never by the controller.

| ID | Test | Pass criterion |
| --- | --- | --- |
| T1 | Standard golden trace | Bitwise equal to the `1ff631bb` fixture |
| T2 | Extraction lemma | `Θ_k` from the superset equals four independent QR-RLS fits on `φ_k` (relative 1e-6), with forgetting and prior |
| T3 | One update per genuine WFS | 1,000 frames give exactly 1,000 multirate updates, 4,000 extractions, 4,000 predictions and 0 repeated targets |
| T4 | Intermediate ACC acts on the next slot | With frozen gains, a step in `a_{c_i+2}` changes `u_{i,2}` by `K_{2,2}`·Δ. `u_{i,0}` and `u_{i,1}` are unchanged (causality). |
| T5 | No-information invariance | With frozen `Θ_k = Θ_0` and no probes, the 8 kHz commands equal the 2 kHz plan with the prefix fixed (1e-9) |
| T6 | Skipped slot | A missed copy enters the prefix and the lift as the retained value; the next slot compensates |
| T7 | WFS late (`p ≥ 4`) | Fallback plan entries are published and counted; no fault |
| T8 | ACC late (`k < p`) | `K_{k,p}` is used and counted |
| T9 | Closed-loop simulation, strong/weak/zero ACC information (SNR ∞-ish / 1 / independent ACC) | Strong: residual variance well below standard (target < 0.7×). Weak: at most standard. Zero: within 1.1× standard. Report per-phase RMSE, frame and 8 kHz residual variance, and PSDs against standard and 2 kHz assisted |
| T10 | Mode switching, learning, prediction and removal through INDI callbacks | Existing control tests extended to the fast path |
| T11 | Sanitizers | ASan/UBSan/TSan clean on the controller and output suites |
| B1 | Benchmark at RTC dimensions | Fast tick mean/p99/max, learning burst mean/p99/max. Mac, then RTC run by the user. Pass: fast p99 < 10 µs; learning p99 < 400 µs per frame |

On the RTC (user-run commands, added to `RTC_TESTING.md`), a 60 s capture with prediction
on is expected to show:

- `fast_predictions` and `transport_published` rising by about 480 k;
- `multirate_updates` rising by about 120 k while learning;
- `extractions` rising by about 360 k while learning;
- `transport_skipped`, `fast_fallbacks`, `learn_overruns` and `acc_phase_anomalies` near 0;
- `phase_hist` concentrated at `k−p = 0`.

A/B: alternate frozen standard and assisted fits under the same accelSim disturbance, and
compare WFS residual variance and PSDs.

## 8. Risks

- Closed-loop bias in `C` (§2.5 item 7) is quantified by T9.
- Pre-flight on the RTC: CPU budget, both B1 numbers.
- The ACC sample may arrive after the slot render, so `k < p`. D4 and the histogram
  address this.
- The ACC window is short compared with low-frequency vibrations. Low-frequency position
  still comes from `y_p`. Lengthen `accel_history` if RMSE shows benefit.

## 9. Review decisions (2026-11-20)

- **Approved:** D1–D3 and the shared QR-RLS formulation, subject to the T2 numerical check.
- **D4:** use the newest ACC sample. Measure its age at each calculation and at DM application, and keep reliable
  scheduler deadlines.
- **Standard training:** the standard model trains only while standard is selected, so the two controllers are
  trained independently.
- **Standard identity:** standard mode must be bit-for-bit identical to `1ff631bb` (T1), with no timing regression.
- **Ratios:** any integer ACC:WFS ratio and DM slot count within the scheduler's capabilities.
- **Updates:** the prediction thread receives coherent model updates without blocking on learning.
- **Scope:** nothing is deployed or activated on the instrument.

## 10. Implementation (uncommitted, branch `parkertjohnson1/accel_ddspc`)

### `multirate_ddspc.hpp/.cpp` (new)

- **`InformationQR`:** the augmented factor `[R | Z]`, updated by row Givens rotations in double precision. The
  `solve(leading)` call returns `R_SS⁻¹ Z_S`. Each rotation touches only its own and later columns, so the leading
  block is the leading-subset problem exactly, not just in exact arithmetic. T2 measures zero difference.
- **`MultirateDDSPC`:** frame ring, counter-keyed ACC ring, one matured example per WFS frame and one QR update
  while training. The a priori RMSE of each phase is scored before the update. `refresh()` extracts the four
  `Θ_k` and computes the `r·q` slot gains.
- **`slot_gain`:** forms only the next slot's rows, `−(C_v H⁻¹E)ᵀ[M | C_pre] − (H⁻¹E)ᵀ μ P_v,pre`. That is about
  8 k multiply–adds per (k,p), with no `C_vᵀM` product.
- **`command()`:** one m-row multiply.

### `loPredCtrlAcc_operational.hpp`

**Standard path.** `step()` is unchanged apart from `c_slots`/`c_ring` becoming the members `m_slots`/`m_ring`
(both still 4 and 16). Assisted frames branch to `stepAssisted()` before any standard code runs.
`outputCandidate()` sends assisted requests to `assistedCandidate()`.

**`stepAssisted` (WFS thread):**

1. The frozen standard fit's histories keep running, so reselecting standard causes no history gap.
2. `MultirateDDSPC::append`.
3. Under `m_commandMutex`: publish the origin, base, feedback and probe strength.
4. Then `learn()` and `refresh()` into the back gain buffer, and a swap with the middle buffer.

**`assistedCandidate` (DM worker):**

1. Sleep, then spin, until `cutoff − renderLeadUs`.
2. Swap in fresh gains (the front buffer is owned by the worker) and copy a new origin.
3. Build the executed prefix from the scheduler copies; a slot that was not copied takes the retained value.
4. Read new ACC samples through the worker's own `AdcReader` cursor.
5. Compute `k` from the anchor and calculate one command.
6. Probes are per slot. A slot with no current frame (late WFS) holds the last command.
7. Values beyond float range are zeroed.

**Training.** `applyTraining()`: standard learns only when selected. The shadow assisted model is never trained.
`m_assistedCommandSlots = 1` makes it dormant and removes the four-slot limit.

**Removed.** The matched-pair sink, which needs two simultaneously trained models.

**New configuration:** `parameters.dm_slots`, `parameters.acc_ratio` (1–16), `parameters.ripple` (default 10) and
`dm.renderLeadUs` (default 40).

**New INDI:** rates, latency distributions (1-µs histogram per INDI interval), holds, lagging, anomalies,
mismatch, per-phase RMSE and ACC-phase counts, learning time and overruns.

### Ripple penalty: a deviation from the approved cost, needing review

The approved cost penalizes only frame-integrated WFS residuals. In closed-loop simulation the lifted per-slot law
(at `1ff631bb` and here) injected within-frame ripple:

| Case | Assisted | Standard |
| --- | --- | --- |
| ACC disconnected | 1.63 | 0.70 |
| ACC is independent noise | 1.79 | 0.98 |

Most of the excess is above the WFS Nyquist frequency and invisible to the WFS. I added `μ‖P u_f‖²`, where `P`
removes each future frame's mean command per mode. It is scaled by `max(CᵀC)` like `λ` and is independent of the
`λ` exploration stages. `ripple=0` is the original cost.

T5 shows the slot re-solve is still exact (6e-16). The value was chosen from a sweep over μ = 1, 10, 100 and 1000:

| μ | Strong | Weak | Zero | Disconnected |
| --- | --- | --- | --- | --- |
| 1 | 0.42 | 0.79 | 1.52 | 1.42 |
| 10 | 0.17 | 0.46 | 1.27 | 1.20 |
| 100 | 0.24 | 0.38 | 1.33 | 0.76 |

At these values the standard results were 0.98 for strong, weak and zero, and 0.70 for disconnected. The default
is 10 (the λ-independent form of μ = 100 at the final `λ = 0.1` stage).

## 11. Results (development Mac, offline; RTC validation pending)

### T1 standard equivalence

`tests/standard_equivalence.py` exports `1ff631bb`, builds both trees with the same compiler and flags, and runs
three scenarios three times each:

- RTC dimensions with ACC present, using the original exploration sequence;
- RTC dimensions with missed copies, a WFS gap and a gain change;
- conventional RLS.

**Result:** PASS. All 192,000 rendered values are bit-for-bit identical. Standard decision time:

| Scenario | 1ff631bb | Now |
| --- | --- | --- |
| Learning, ACC present | 136 µs | 13 µs (the shadow assisted fit no longer trains alongside) |
| Frozen | 2.9 µs | 1.3 µs |
| Edge scenario, learning | 14.3 µs | 13.5 µs |
| Edge scenario, frozen | 2.3 µs | 1.5 µs |

### T2 shared-factor equivalence

| Check | Shared vs four independent QR-RLS | Shared vs batch least squares |
| --- | --- | --- |
| RTC learner, 212 columns | 0 (bit-identical) | 6.7e-13 relative |
| Core test, γ = 1 | 0 | 5.9e-15 |
| Core test, γ = 0.995 | 0 | 1.1e-14 |

### Rates, causality and command history (T3–T8)

- **T3:** 100 frames gave exactly 100 QR updates and 400 slot predictions. Equal counts at k = 0..3. Consecutive
  targets are distinct frames.
- **T4:** changing the slot-2 ACC sample changes slot 2 by exactly `K(2,2)`·Δ. Earlier slots were already
  rendered.
- **T6:** a slot that was not copied enters the prefix as the retained copy.
- **T7:** a late WFS frame causes a hold.
- **T8:** a late ACC sample means k < p and is counted.

### T9 closed loop

The plant has narrowband AR(2) vibrations at 130, 310 and 470 Hz, a slow WFS-only drift and a WFS integrating four
slots. Each mode was trained with 4× the loPredCtrl sequence (12,000 frames) and frozen for 3,000 frames.

| ACC information | Residual var., standard | Residual var., assisted | Prediction RMSE, standard | Prediction RMSE, assisted k = 0 → 3 |
| --- | --- | --- | --- | --- |
| strong (noise 0.02) | 1.177 | 0.236 | 0.376 | 0.140 → 0.115 |
| weak (noise 1) | 1.177 | 0.464 | 0.376 | 0.283 → 0.258 |
| zero (independent) | 1.177 | 1.238 | 0.376 | 0.388 → 0.397 |
| disconnected (0) | 1.087 | 1.199 | 0.369 | 0.377 |

The open-loop variance is about 1.9. `tests/multirate_report.py` computes the PSDs and band RMS. With strong ACC
information:

- RMS fell in every band below 1 kHz (400–1000 Hz: 0.83 → 0.09);
- RMS rose above 1 kHz (0.21 → 0.45), the slot-rate ripple that the WFS cannot see.

### B1 benchmark (RTC dimensions)

| Measurement | Mean | p99 | Max |
| --- | --- | --- | --- |
| Assisted WFS frame while learning | 158–164 µs | 182–333 µs | |
| Assisted WFS frame, frozen | 1.8 µs | | |
| Slot calculation | 0.2–0.5 µs | 2 µs | |

**Concurrent run:** 8,000 WFS frames at 2 kHz with learning, alongside 32,000 slots at real 125-µs cutoffs and
renderLead 40.

- One QR update per WFS frame: 7,982 updates.
- 31,037 slot predictions.
- 0 history breaks.
- Median margin 37 µs; 0.6% of cutoffs missed, with a minimum margin of −1.2 ms. This is macOS without real-time
  priority and must be repeated on the RTC.

### Sanitizers

ASan+UBSan and TSan are clean, including the concurrent benchmark under TSan.

## 12. Remaining before commissioning

- Review the ripple penalty and its default.
- Run the benchmark on the RTC on spare28/29, then measure the ACC age, margin and holds through INDI. Tune
  `renderLeadUs`.
- ACC anchor rule D1 relies on the upstream 4:1 lock. Watch `acc_anomalies` and the equal `acc_phase` counts.
- T9 uses one realization per case. Results vary by about 10%, so the zero-information margin (≤ 1.2× standard)
  is not a guarantee.
- Closed-loop identification bias was not measured separately. It is mitigated by per-slot exploration and frozen
  evaluation.
- The pre-existing `native_linux_build_test` failure (default shadow-only expectation) is still out of scope.

## 13. RTC blocker: `accel_sync=false` with `delay_lock=0` (2026-11-20)

### Observed upstream state

| Parameter | Value |
| --- | --- |
| ACC rate | 8 kHz |
| WFS period | 500 µs |
| `mode_code` | 1 |
| `delay_lock` | 0 |
| `delay_applied` = `delay_model` | 409.4 µs |
| `delay_phase_error` | 0 |
| `read_latency_error` | ≈ 52 µs |
| Drops and gaps | none |

### Analysis (`accelSim.hpp`)

Before this change, `delay_lock` had two writers:

- **`syncTiming()` at each WFS reference.** It called `updateSyncDiagnostics()`, which set lock from the phase
  thresholds alone.
- **`acquireAndCheckValid()` at each 8 kHz sample.** It cleared lock when that one sample's lateness exceeded
  50 µs, or 10% of the WFS period.

The INDI snapshot is taken after each publication, so it always reflects the last acquisition. A phase error of
0 passes both thresholds, so lock = 0 can only come from acquisition lateness above 50 µs.

`read_latency_error_us` is the lateness of the sample taken at the reference's predicted phase. It reads ≈ 52 µs,
so the samples really are about 52 µs late. That is a genuine acquisition-timing failure. The phase-tracking clock
is fine.

### Cause

In synchronized mode the worker reaches every deadline through a `sem_timedwait` timeout, which is a futex wait.
Futex timeouts carry the thread's timer slack, 50 µs by default, and older kernels apply it to SCHED_FIFO threads
too. Timer mode uses `clock_nanosleep`, which is exempt for real-time threads. That predicts 50 µs plus a few µs of
service: the 52 µs observed. This is inferred from the code and the numbers; RTC confirmation is below.

### Fixes (thresholds unchanged)

- **Timer slack:** `startAcquisition()` sets the acquisition thread's timer slack to 1 ns with
  `prctl(PR_SET_TIMERSLACK, 1)`; 0 would restore the default.
- **Coherent lock:** `diagnostics::updateLock()` applies the unchanged absolute and fractional thresholds to:
  - the phase error;
  - the largest acquisition lateness over the current and previous WFS periods.

  It is evaluated the same way at references and at acquisitions. A reference can no longer re-assert lock over late
  samples, and one complete on-time period restores it.
- **New `sim_diag` elements:** `lateness_last_us`, `lateness_window_max_us`, `lock_phase_ok`, `lock_timing_ok`,
  `acq_timer_slack_ns`, `acq_sched_policy` and `acq_priority`.

**Tests.** The model-level test "accelSim delay lock window" passes, as do 19 model/sync tests and the 27 DM/timeline
tests (which compile `accelSim.hpp`). The app-level test "accelSim delay lock qualifies phase and acquisition
lateness coherently" needs the native build; it was not run here.

**RTC confirmation (operator):**

- Before installing: `cat /proc/<acq tid>/timerslack_ns` should read 50000.
- After installing:
  - `acq_timer_slack_ns` should read 1;
  - `read_latency_error_us` and `lateness_window_max_us` should fall well below 50;
  - `delay_lock` should read 1.

### loPredCtrlAcc

- **`assisted_qualification` property:** `blockers` lists every failing condition, and `assisted_state` reports the
  first. It also shows:
  - the upstream values;
  - the reason the ACC window failed (`sensorHistory` now records it);
  - the model's update count;
  - the diagnostic mode.
- **`parameters.assisted_diagnostic` (default false):** lets the learner use complete ACC windows while unlocked.
  - Output still requires the lock, through `outputValid` and `m_syncLocked`. The DM transport is unchanged.
  - No physical probes are applied while unlocked.
  - Learning while unlocked sets `m_unlockedTraining`. That refuses assisted prediction and clears candidate validity
    until `reset_model`.
  - Frozen evaluation (`rmse_phase*`) does not set the flag.
- **Standard mode:** standard paths are unchanged; T1 is re-run.

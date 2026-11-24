# loPredCtrlAcc laboratory operation

The normal build runs closed-loop DDSPC on the existing WFS monitor (`aol1_modevalWFS`, 2 kHz). It publishes a full calibrated surface through the syncopation producer to its own auxiliary channel, `dm.channelName`.

- **Standard mode** uses loPredCtrl's DDSPC calculation: one decision and RLS update per WFS frame on the WFS thread, held on that frame's four DM slots. It does not use the accelerometers and runs with `adc.enabled=false`. loPredCtrl's commands were integrated by the main loop's mfilt; here they are absolute positions unless the optional integrator, which also covers assisted mode, is enabled (see "Integrator").
- **Assisted mode** recalculates every 8 kHz DM slot on the DM worker from the newest upstream ACC sample, the newest WFS frame and the commands the scheduler actually copied. Learning is one shared QR-RLS update per genuine WFS frame (2 kHz) on the WFS thread; the four phase predictors and their slot gains reach the worker through a triple buffer, so the worker never waits for learning. The formulation and its validation are in `agents/plans/2026-11/ddspc-multirate-8khz-assisted.md`.

Every WFS frame is also republished to the main AO loop. With prediction off it is unchanged; with prediction on, tip/tilt is set to zero, so DDSPC is the only tip/tilt controller (see "Main-loop modal output").

Each model trains only while its mode is selected, so the standard and assisted fits are independent experiments. Both learn only from genuine WFS measurements and the commands the scheduler copied from this channel.

## DM channel, worker and timing

| Setting | Value | Reason |
| --- | --- | --- |
| `dm.channelName` | `dm01disp09_aux` | One producer per scheduler destination. The `<aux>.producer` lock is exclusive, and synco copies the whole image. `acc000` owns `dm01disp10_aux`, so the disturbance and the correction must use different destinations; dmcomb sums them. The code accepts any `*_aux` name. |
| `dm.threadPrio` | `20` | Dedicated producer worker started by `MagAOXApp::threadStart`, as in accelSim and dmSpeckle. It runs below the scheduler's priority 30. |
| `dm.cpuset` | `spare29` | Dedicated development core (CPU 29). acc000 uses `spare30` and `spare31`. |
| `shmimMonitor.cpuset` | `spare28` | Runs the WFS decisions and all learning. Assisted learning takes about 160 µs per 500 µs frame on a development Mac (one 212-column QR update, four phase solutions, 16 slot gains). |
| `dm.renderLeadUs` | `40` | Assisted only. Each slot is calculated at `cutoff - renderLeadUs` with the newest ACC sample; the calculation takes about 1 µs. Standard slots are still rendered on eligibility. |
| `syncopation.writeLeadUs` | `125` | Publishes as soon as a slot is rendered. |
| `parameters.dm_slots`, `acc_ratio` | `4`, `4` | DM slots and ACC samples per WFS frame. Other integer ratios are supported (1–16) when the scheduler and upstream sampling provide them. |
| `parameters.ripple` | `10` | Assisted only; see below. |

The worker polls every 10 µs and checks the next slot immediately after a commit. A slot that cannot be committed by its cutoff is counted in `ddspc_performance.transport_skipped`; synco then keeps the last copied contribution. Such a slot no longer latches output off.

### Slot mapping

A slot ordinal is the scheduler `cycle*steps+step`. A WFS frame owns the q slots after the newest slot copied when it arrived. Training takes the q slots copied since the previous decision, oldest first; a missed slot holds the retained copy. The standard model receives their mean, which equals its held command.

In assisted mode, slot p of a frame is calculated with:

- the frame's WFS measurement and the WFS/command histories, published before learning starts;
- the k ACC samples received since the frame's anchor (the newest ACC counter at the WFS decision; upstream owns the 4:1 lock), normally k = p, fewer if a sample is late (`acc_lagging`);
- the p commands already executed in the frame: the scheduler's copies, or the retained copy for a slot that was not copied.

A slot with no current frame (a late WFS frame) holds the last command (`fast_holds`). `ddspc_performance.slot_shifts` counts decisions whose interval contained other than q new copies; a steadily rising count means the WFS and DM clocks are not locked.

### Within-frame ripple

The DDSPC cost penalizes future WFS frames, which integrate q slots, so command variation within a frame is unobserved. In simulation the unmodified cost (`ripple=0`) added within-frame ripple that made assisted control worse than standard when the ACC carried no information. `ripple` penalizes each future frame's variation about its mean command, scaled like `regularization`; with `ripple=10` and no ACC information assisted stayed within 5–10% of standard, and with informative ACC it reduced the 8 kHz residual variance by 60–80%. Re-solving with fixed executed slots remains exact for this cost (test T5). Measure the WFS residual and, if available, an independent fast sensor when tuning it.

### Integrator

The original loPredCtrl wrote its command into the main loop's modal input, so cacao `mfilt` integrated it: `DM[k] = mult·DM[k-1] − gain·delta[k]`, clipped to ±limit. Its model was trained on those deltas, and the main loop kept integral action on tip/tilt, which tweeter-to-woofer offloading relies on. By default loPredCtrlAcc applies the same DDSPC calculation as an absolute position, with no integrator, and `parameters.gain·y` is proportional.

`[integrator] enabled=true` restores the mfilt law in both modes, with the main loop's own parameters. The output still goes through syncopation to `dm.channelName`; only the value written changes. Each command `delta` (including `gain·y` and probes) updates `s = mult·s − gain·delta`, clipped to ±`limit`, and synco writes `s`:

- **Standard:** one update per WFS frame, held on the frame's four slots, exactly as mfilt. A late WFS frame holds `s`.
- **Assisted:** one update per DM slot on the DM worker with `gain/dm_slots`; `mult` and `limit` are unchanged. A slot with no current frame holds `s`.

For each DDSPC mode `i` (WFS index `modal_index0/1`), loPredCtrlAcc follows the main loop as it changes:

| Parameter | Source |
| --- | --- |
| `gain` | `<loop.device>.loop_gain.current × aol<N>_mgainfact[i]` |
| `mult` | `<loop.device>.loop_multcoeff.current × aol<N>_mmultfact[i]` |
| `limit` | `<loop.device>.loop_max_limit.current × aol<N>_mlimitfact[i]` |

`loop.device` defaults to `holoop` and `loop.number` (N) to `1`. The combined values are published as `loPredCtrlAcc.integrator.*` (`enabled`, `ready`, `gain0/1`, `mult0/1`, `limit0/1`). Prediction is refused, and `safety.output_state` reports `integrator_parameters`, until all six have been received with positive gain and limit for both modes. After that each keeps its last value until it is updated.

Each fit is trained on the deltas recovered from the copied commands, `−(p − mult·p_previous)/gain` per frame (standard) or per slot with `gain/dm_slots` (assisted), so missed slots and clipping are seen as applied. The assisted executed-slot prefix uses the same deltas, and `ripple` then penalizes delta variation within a frame. The state is zero whenever prediction is off, so prediction always starts from the confirmed zero; a fault zero still removes the whole state in one slot. A change of the main loop's gain changes the plant a frozen fit was trained on; keep learning on, or retrain, after one.

```bash
getINDI 'loPredCtrlAcc.integrator.*' 'holoop.loop_gain.current' 'holoop.loop_multcoeff.current' 'holoop.loop_max_limit.current'
```

The feedback sign matches mfilt only if `dm.modeCube` planes 0 and 1 use mfilt's DM mode sign. It is a startup setting; the restart that changes it also discards the fits, which would not apply in the other form. The design is in `agents/plans/2026-11/ddspc-standard-integrator.md`.

## Before installation (operator, on the RTC)

```bash
hostname   # the RTC
grep -n destinations /opt/MagAOX/config/dmtweeter-synco.conf   # must list dm01disp09
ls -l /milk/shm/dm01disp09_aux.im.shm                          # must exist with the tweeter geometry
lslocks | grep -E 'dm01disp09_aux' || echo 'no producer owns dm01disp09_aux'
cat "$CGROUPS1_CPUSET_MOUNTPOINT/spare28/tasks" "$CGROUPS1_CPUSET_MOUNTPOINT/spare29/tasks"   # both empty
```

If `dm01disp09` is not a scheduler destination or is in use, choose another unused destination and set `dm.channelName` to it. Changing the scheduler configuration is a separate decision.

## Build and install

```bash
cd ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc
make -C ../.. flatlogs/bin/flatlogcodes
make -j4 DDSPC_OPERATIONAL_BUILD=yes all
rtc_config=$(mktemp --suffix=.conf)
cp loPredCtrlAcc.conf "$rtc_config"
diff -u /opt/MagAOX/config/loPredCtrlAcc.conf "$rtc_config"
"${EDITOR:-vi}" "$rtc_config"   # carry over working WFS, calibration and ADC settings if they differ
xctrl stop loPredCtrlAcc
make -j1 DDSPC_OPERATIONAL_BUILD=yes install
sudo install -m 0644 --backup=numbered "$rtc_config" "$(readlink -f /opt/MagAOX/config/loPredCtrlAcc.conf)"
xctrl start loPredCtrlAcc
```

`make install` keeps an existing `/opt/MagAOX/config/loPredCtrlAcc.conf`; the explicit `install` line replaces it. accelSim (`acc000`) keeps its own configuration: `[dm] threadPrio=20`, `cpuset=spare31`, `channelName=dm01disp10_aux`, `[syncopation] writeLeadUs=125`.

The DM basis comes from `dm.modeCube`, planes 0 and 1 at their stored scale. Confirm that `aol1_modevalWFS` entries 0 and 1 correspond to those planes in sign and units on this routing. Otherwise supply the deployed mapping with `dm.calibration` (DDSPC_MODAL_V1, [SYNCO_NATIVE.md](SYNCO_NATIVE.md)). For an AD7689, keep the working ADC calibration and set `adc.sync_device` to its INDI name and `adc.sync_property=ddspcTiming`.

## Startup checks

```bash
getINDI 'loPredCtrlAcc.th-dm.*'                  # priority 20
getINDI 'loPredCtrlAcc.safety.*' 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.adc_values.*'
getINDI 'acc000.timingDiag.mode_code' 'acc000.timingDiag.delay_lock'   # both 1 for assisted mode
```

Wait for `safety.zero_confirmed=true`, the consumer's receipt for the initial own-channel zero. If it stays false, check the scheduler, the channel and `safety.output_state`. If a readiness flag stays false, `ddspc_control.standard_state` and `assisted_state` give the reason. Possible reasons are an absent WFS, the upstream lock, sensor history, insufficient training or excitation, and numerical faults. A refused `predict` or `learn_std` request is answered at once with its unchanged state and Alert; `predict` stays Alert until a request is accepted, which `loPredCtrlAccGUI` shows as a warning.

Start the disturbance and record a 60 s baseline with prediction off:

```bash
setINDI -n 'acc000.dm_scale.target=<approved scale>'
setINDI -s 'acc000.dm_enable.toggle=On'
```

## Assisted qualification

`ddspc_control.assisted_state` reports the first failing condition. `assisted_qualification.blockers` lists all of them, for example `acc000.timingDiag.delay_lock=0,model_untrained_0_of_212_updates`; `none` means assisted mode is qualified. The other elements give the values behind them: `upstream_enabled` and `upstream_locked` (`mode_code` and `delay_lock`, or `enabled` and `valid` for `ddspcTiming`), `upstream_phase_error_us`, `upstream_read_latency_error_us`, `upstream_age_s`, `adc_state`, `acc_window` (the WFS thread's ACC window: `ok`, `history_filling`, `stale_sample`, `counter_gap`, ...), `model` (updates/needed) and `diagnostic_mode`.

```bash
getINDI 'loPredCtrlAcc.assisted_qualification.*'
getINDI 'acc000.timingDiag.delay_lock' 'acc000.timingDiag.delay_phase_error_us' 'acc000.timingDiag.read_latency_error_us'
getINDI 'acc000.sim_diag.lock_phase_ok' 'acc000.sim_diag.lock_timing_ok' 'acc000.sim_diag.lateness_window_max_us' \
        'acc000.sim_diag.acq_timer_slack_ns' 'acc000.sim_diag.acq_sched_policy' 'acc000.sim_diag.acq_priority'
```

With accelSim, `delay_lock=0` while `delay_phase_error_us=0` means late acquisitions (`lock_timing_ok=0`); see the accelSim README. After installing the updated accelSim, `acq_timer_slack_ns` should read 1, and `read_latency_error_us` and `lateness_window_max_us` should fall well below 50.

### Diagnostic mode (default off)

`parameters.assisted_diagnostic=true` lets the assisted learner append, learn and score complete ACC windows while the upstream lock is down, so the ACC model can be evaluated during commissioning. It never publishes physical corrections: assisted output still requires the upstream lock, and nothing about the DM transport changes.

- To evaluate an assisted fit trained with lock, keep learning off and read `rmse_phase0..3`.
- With learning on, the fit learns without lock. `diagnostic_mode` then reads `on_learned_without_lock`, and prediction is refused until `reset_model`.
- Exploration probes are physical commands, so they are not applied while unlocked; the learner sees the commands actually copied.

## Train and test standard

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=On;accel_assisted=Off'
setINDI -n 'loPredCtrlAcc.gain.target=0.3'
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.safety.*'
```

The triples are loPredCtrl's original sequence: WFS frames, Gaussian standard deviation in modal command units, and regularization. Use the middle value that matches the deployed modal scale. `setINDI -s 'loPredCtrlAcc.learn_std.toggle=On'` installs `1000,0.05,100,1000,0.05,3,1000,0.05,0.1` and enables learning and prediction together. During the 3,000-frame sequence (1.5 s), `prediction=warming` shows probe or feedback commands. Learning continues after the sequence ends, and the final stage's regularization (0.1) stays in use until `reset_model`. When `standard_ready=true`, freeze the fit and measure:

```bash
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
# measure 60 s with the standard correction applied
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # wait for zero_confirmed=true before changing mode
```

## Train and test assisted

Select assisted and train it; only the assisted fit learns, and the standard fit stays frozen. Probes are drawn independently for every slot, which identifies each slot's response.

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=On'
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*'
# when assisted_ready=true:
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
# measure 60 s with the assisted correction applied
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
```

For an A/B comparison, train each mode while it is selected, keep the disturbance unchanged, then alternate the frozen fits. Switch modes only after `zero_confirmed=true`. No reset is needed. `standard_rmse` is the standard fit's origin forecast RMSE; `rmse_phase0..3` are the assisted phase predictors' a priori future-WFS RMSE (k = 0..3 new ACC samples) and should decrease with k when the ACC is informative. Optical correction is judged from the WFS residual with prediction off and on.

## 60 s publication check

Take `getINDI 'loPredCtrlAcc.ddspc_performance.*'` and `getINDI 'dmtweeter-synco.production.*'` 60 s apart with prediction active. Expect:

- `transport_published` and `transport_acks` to rise by about 480,000, and `publish_rate_hz` near 8000.
- `transport_skipped` and `slot_shifts` to stay near zero.
- In assisted mode: `predict_rate_hz` near 8000 (a calculation for every slot, not a replayed plan); while learning, `update_rate_hz` and `refresh_rate_hz` near 2000 (one QR-RLS update per WFS frame); `fast_holds`, `acc_lagging`, `acc_anomalies`, `step_mismatch` and `learn_overruns` near zero; `acc_phase0..3` equal; `margin_us_min` positive; `learn_us_p99` well below 500.
- `acc_age_calc_us_*` and `acc_age_apply_us_*` give the age of the newest ACC sample at each calculation and at the slot's DM target time. Lower `dm.renderLeadUs` while `margin_us_min` stays positive with headroom; raise it if `transport_skipped` rises.
- `safety.output_state` to report `transport_acknowledged_not_physical_application`.
- The scheduler's accepted count for the `dm01disp09` index (the third destination, `accepted2`) to rise by about 480,000, with `missing2` and `rejected2` near zero.

## Benchmark on the RTC (optional, operator decision)

The offline benchmark runs the operational code at RTC dimensions: sequential standard and assisted timing, then a WFS thread learning at 2 kHz concurrently with a DM worker calculating every 125 µs slot. It uses no instrument stream. Run it on a spare core only:

```bash
cd ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc
make -f tests/Makefile.offline check-rtc-controller
taskset -c 28,29 build/offline/rtc-controller/ddspc_offline_test '[rtc-benchmark]'
```

## Main-loop modal output

loPredCtrlAcc always republishes every new `aol1_modevalWFS` frame to the main AO loop, as the original loPredCtrl does. What it publishes depends on who owns tip/tilt:

| DDSPC state | Main-loop modes 0 and 1 | Other modes | `ddspc_control.main_loop_tt` |
| --- | --- | --- | --- |
| Prediction off (including at startup) | unchanged | unchanged | `passthrough` |
| Prediction on (`predict` or `learn_std` accepted) | zero | unchanged | `zeroed` |
| Prediction turned off or faulted, own-channel zero not yet confirmed | zero | unchanged | `zeroed` |

With prediction off the main loop sees exactly the WFS frame, as if loPredCtrlAcc were not there. From the moment a prediction request is accepted, tip/tilt is zero on the next WFS frame and DDSPC, through its 8 kHz auxiliary channel, is the only tip/tilt controller. After prediction stops, tip/tilt stays zero until `safety.zero_confirmed=true`, so the main loop and DDSPC never both correct tip/tilt. If that zero is never confirmed (for example, the scheduler stopped), tip/tilt stays zero until it is, until the operator sends `release_tt`, or until loPredCtrlAcc restarts. The startup zero of the auxiliary channel does not withhold tip/tilt. Every other mode is always copied bit-for-bit. Nothing else changes:
- DDSPC still learns, predicts and applies feedback from the unmodified `aol1_modevalWFS` tip/tilt.
- The input stream is never written.
- Learning, predictive gains, ACC handling, scheduler timing and the auxiliary channel are unchanged.

| Setting | Default | Meaning |
| --- | --- | --- |
| `framegrabber.shmimName` | app name | Output stream, as loPredCtrl's `[framegrabber] shmimName`. Use the stream the main loop's modal filter reads. Startup rejects the WFS input, a `*_aux` channel, the scheduler timing stream and an empty value. |
| `framegrabber.threadPrio`, `cpuset` | 2, none | Publisher thread, as loPredCtrl. Copy them from the deployed loPredCtrl.conf. |
| `parameters.own_shmim` | `false` | As loPredCtrl, but its default there is true. `true` destroys the stream at shutdown; `false` leaves it for the main loop. |

Startup requires `num_modes=2` with `modal_index0=0` and `modal_index1=1`, so the zeroed modes are exactly the DDSPC-controlled ones.

**One publisher only.** The publisher takes an exclusive lock, `<stream file>.producer`, the convention of the syncopation producers. It creates the stream only if it is absent. An existing stream is attached unchanged, so the main loop keeps its mapping, but only when it has the same shape and float type and its counter does not advance for 20 ms. The original loPredCtrl takes no lock; the counter check is what refuses its stream while it is still publishing. A refusal is logged once and retried every second. Stop loPredCtrl before starting loPredCtrlAcc, and never run both.

**What the main loop does with zero tip/tilt.** The deployed modal filter is cacao `mfilt-1` (`agents/plans/2026-10/rtc-cpuset-restructuring.md`; its source is not in this repository). The MagAO-X `modalFilter` app implements the same kind of law, `DM[k] = -g*gfact*WFS[k] + mc*mfact*DM[k-1]` (`apps/modalFilter/modalFilter.hpp:997-998`). With zero input, each tip/tilt state holds its last value when `mc*mfact = 1`, or decays geometrically when it is below 1. **It is not cleared.** The tip/tilt command the main loop held when prediction was turned on therefore stays on its DM channel, summed with the DDSPC contribution, and the main loop resumes from it when prediction is turned off. Opening the main loop is a different case: it zeroes its own state (modalFilter) and stops all modes. loPredCtrlAcc never changes main-loop gains, multipliers or loop state.

Before installing, check these on the RTC:
1. Which stream the main loop's modal filter (`mfilt-1`) reads, and that it is not `aol1_modevalWFS`. Use loPredCtrl's `[framegrabber] shmimName` if loPredCtrl feeds it, and set the same name here. If the filter reads `aol1_modevalWFS` directly, nothing reaches the main loop until it is repointed, which is a separate cacao configuration decision.
2. That `aol1_modevalWFS` is published upstream by the reconstructor and nothing writes the output stream back into it, so DDSPC never trains on zeroed tip/tilt.
3. Who publishes the output stream now: stop loPredCtrl and confirm its counter stops advancing.
4. The main loop's tip/tilt gain and multiplier factors (`mgainfact`/`mmultfact` or the cacao equivalents), to know whether the held state is constant or leaking.

Handover, with every step an operator decision:
1. With the main loop closed, stop loPredCtrl.
2. Start loPredCtrlAcc with prediction off. The main loop receives unchanged frames and keeps correcting tip/tilt (`main_loop_tt=passthrough`).
3. Train and enable DDSPC as above, with prediction on (`learn_std`, or `learn` with `predict`). From the accepted request, the main loop's tip/tilt input is zero (`main_loop_tt=zeroed`) and its tip/tilt state holds (or slowly leaks), so the DM does not jump. DDSPC learns the plant with the main loop no longer correcting tip/tilt; learning with prediction off sees the main loop's tip/tilt correction instead, a different plant.
4. Leave the held tip/tilt state in place unless it must be removed. It is a constant offset on the DM, and it supplies the static tip/tilt that DDSPC's proportional feedback (`gain*y`) may not hold by itself. To remove it smoothly, lower only the tip/tilt multiplier slightly below 1 (about 0.999 gives a 0.5 s time constant at 2 kHz) and restore it after the offset has decayed. Never zero the multiplier or reset the loop: that removes the whole offset in one frame.

To hand tip/tilt back, turn prediction off. Once `zero_confirmed=true`, `main_loop_tt` returns to `passthrough` and the main loop resumes tip/tilt correction from its held state. loPredCtrlAcc can keep running; stop it and restore loPredCtrl only to remove it from the main loop's input path.

The main loop's modal input depends on loPredCtrlAcc, as it did on loPredCtrl: if loPredCtrlAcc stops, the main loop stops receiving frames for every mode. The output carries the WFS frame's acquisition time. `fps()` is 0, so `fg_timing` latency statistics stay empty. Check publication with `getINDI 'loPredCtrlAcc.fg_shmimName.*' 'loPredCtrlAcc.fg_frameSize.*'` and the stream's counter, which advances once per WFS frame. To confirm what the main loop receives, read one frame of the output stream: with prediction on, entries 0 and 1 are zero and the rest match the same `aol1_modevalWFS` frame; with prediction off, every entry matches.

## Remove the contribution

```bash
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # zero_confirmed=true
setINDI -s 'acc000.dm_enable.toggle=Off'
xctrl stop loPredCtrlAcc
```

Two operator requests recover from a correction left on the DM:

| Request | Effect |
| --- | --- |
| `zero_dm.request` | Turns prediction off and writes a fresh zero to `dm.channelName`, even if the last removal was already confirmed. Main-loop tip/tilt follows the usual rule: zeroed until that zero is confirmed if DDSPC was engaged, otherwise unchanged. |
| `release_tt.request` | Turns prediction off and passes main-loop tip/tilt through from the next WFS frame, without waiting for `zero_confirmed`. The zero is still retried. Use it when the zero cannot be confirmed and the main loop must take tip/tilt back. |

```bash
setINDI -s 'loPredCtrlAcc.zero_dm.request=On'
setINDI -s 'loPredCtrlAcc.release_tt.request=On'
```

Neither touches the main loop's own integrator. A tip/tilt command the main loop held while DDSPC owned tip/tilt stays on its channel until the main loop corrects it, or until the main loop is opened.

Prediction turns off and own-channel zero is requested when any of these occur:

- a discarded receipt;
- a stale WFS stream or a WFS counter gap, which breaks the model history;
- invalid selected sensor data;
- a numerical fault.

`safety.output_state` gives the reason, and keeps it in parentheses after the zero is acknowledged, for example `fault_zero_acknowledged (input_or_model_qualification_lost: acc_window stale_sample)`. Each fault is also logged as "DDSPC output fault, prediction stopped". A lost qualification names the first failed condition: `wfs_invalid`, `command_invalid`, `wfs_stale` (no WFS frame within `input_timeout`), `acc_window` with the ACC window reason, or `upstream_unlocked`. After confirmed removal, prediction can be re-enabled, or the models reset and retrained (`setINDI -s 'loPredCtrlAcc.reset_model.request=On'`). A normal exit also attempts an acknowledged own-channel removal.

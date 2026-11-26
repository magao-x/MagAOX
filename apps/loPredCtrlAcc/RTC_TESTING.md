# loPredCtrlAcc laboratory operation

The normal build runs closed-loop DDSPC on the existing WFS monitor (`aol1_modevalWFS`, 2 kHz). It publishes a full calibrated surface through the syncopation producer to its own auxiliary channel, `dm.channelName`.

- **Standard mode** uses loPredCtrl's DDSPC calculation: one decision and RLS update per WFS frame on the WFS thread, held on that frame's four DM slots. It does not use the accelerometers and runs with `adc.enabled=false`. loPredCtrl's commands were integrated by the main loop's mfilt; here the integrator, on by default and also used in assisted mode, does the same (see "Integrator").
- **Assisted mode** recalculates every 8 kHz DM slot on the DM worker from the newest upstream ACC sample, the newest WFS frame and the commands the scheduler actually copied. Learning is one shared QR-RLS update per genuine WFS frame (2 kHz) on the WFS thread; the four phase predictors and their slot gains reach the worker through a triple buffer, so the worker never waits for learning. The formulation and its validation are in `agents/plans/2026-11/ddspc-multirate-8khz-assisted.md`.

Every WFS frame is also republished to the main AO loop. With prediction off it is unchanged; with prediction on, tip/tilt is set to zero, so DDSPC is the only tip/tilt controller (see "Main-loop modal output").

- **Main-loop route** (`ddspc_mode.main_loop`) runs the standard model without the DM channel, as the original loPredCtrl: its tip/tilt delta goes into the main loop's modal input for mfilt to integrate (see "Main-loop route").
- **mfilt copy** (`ddspc_mode.mfilt_copy_8k` or `mfilt_copy_2k`) runs no DDSPC: the main loop's own tip/tilt integrator, reproduced here, drives `dm.channelName` while main-loop tip/tilt is zero, written on every DM slot (8k) or once per WFS frame (2k). It tests the DM channel path by itself (see "mfilt copy").
- **mfilt clone** (`ddspc_mode.mfilt_clone_8k` or `mfilt_clone_2k`) is the mfilt copy started from `aol1_modevalDM`, so it picks up where mfilt left off, with the DM total equal to mfilt's state; 2k writes only step 0, with the main loop's write, and both report whether step 0 is in sync with it (see "mfilt clone").

Each model trains only while its mode is selected, so the standard and assisted fits are independent experiments. Both learn only from genuine WFS measurements and the commands the scheduler copied from this channel.

## DM channel, worker and timing

| Setting | Value | Reason |
| --- | --- | --- |
| `dm.channelName` | `dm01disp05_aux` | The tweeter channel DDSPC writes. One producer per scheduler destination. The `<aux>.producer` lock is exclusive, and synco copies the whole image. `acc000` owns `dm01disp10_aux`, so the disturbance and the correction must use different destinations; dmcomb sums them. The code accepts any `*_aux` name. |
| `dm.threadPrio` | `20` | Dedicated producer worker started by `MagAOXApp::threadStart`, as in accelSim and dmSpeckle. It runs below the scheduler's priority 30. |
| `dm.cpuset` | `spare29` | Dedicated development core (CPU 29). acc000 uses `spare30` and `spare31`. |
| `shmimMonitor.cpuset` | `spare28` | Runs the WFS decisions and all learning. Assisted learning takes about 160 µs per 500 µs frame on a development Mac (one 212-column QR update, four phase solutions, 16 slot gains). |
| `dm.renderLeadUs` | `40` | Each slot is rendered at `cutoff - renderLeadUs` from the newest WFS decision (assisted: and the newest ACC sample); the calculation takes about 1 µs. Before this, standard and mfilt copy slots were rendered when selected, up to a slot (125 µs) before their cutoff, so a decision completed in between reached the DM a slot late. |
| `dm.receiptTimeoutMs` | `100` | A commit with no scheduler receipt after this long counts as a failed DM write, and the next slot is written. |
| `syncopation.writeLeadUs` | `125` | Publishes as soon as a slot is rendered. |
| `parameters.dm_slots`, `acc_ratio` | `0`, `0` | DM slots and ACC samples per WFS frame, detected while running (see below); 1–16 fixes a value instead. |
| `parameters.ripple` | `10` | Assisted only; see below. |

`dm_slots` follows the DM scheduler's steps per cycle (read from `syncopation.timingStream`), and `acc_ratio` the ACC samples received per WFS frame, rounded, at any ACC rate: the ACC need not be synchronized to the WFS or run at a multiple of its rate (acc000 at 8 kHz with a 2 kHz WFS gives 4, 7 kHz gives 4 or 3, 6 kHz gives 3, an ACC slower than the WFS gives 1). A measurement within 0.6 of the current ratio keeps it, so a rate near a half-integer does not alternate. Each is measured over one-second windows and confirmed by two agreeing windows. A change resizes and resets the models, and waits until prediction is off; it is then applied on the next WFS frame and logged as "DDSPC detected dm_slots ... acc_ratio ...: models reset", so retrain after changing a rate. `rates` shows the applied `dm_slots` and `acc_ratio` and the measured `scheduler_steps` and `acc_per_wfs` (BUSY until both are known). Prediction is refused until `dm_slots` is known (and, in assisted mode, `acc_ratio`), normally about 2 s after the WFS, scheduler and ACC are running; `assisted_qualification.blockers` shows `dm_slots_detecting`, `acc_ratio_detecting`, or `<name>_<value>_applies_when_prediction_off`. `acc_ratio` is limited to 16 and to `128 / accel_history`, so that `acc_ratio × accel_history` fits the 128-sample ACC history. Design: `agents/plans/2026-11/ddspc-detected-rates.md`.

`ddspc_control.prediction` names why an accepted prediction is not writing: `dm_slots_<q>_differ_from_scheduler_steps_<n>` (assisted slots are not written; also logged once as "DDSPC assisted slots not written", fixed by `dm_slots = 0`), or `holding_<condition>` while holding through lost input; otherwise `awaiting_copy` until the scheduler copies the first command, then `active`.

The worker polls every 10 µs and checks the next slot immediately after a commit. A slot that cannot be committed by its cutoff is counted in `ddspc_performance.transport_skipped`; synco then keeps the last copied contribution. Such a slot no longer latches output off.

A failed DM write never stops prediction either. The output stays on, keeps predicting, and writes the next slot; synco keeps the last copied contribution meanwhile. Failed writes are:

- a command the scheduler discards (late for its slot, or stale after a resynchronization), counted in `ddspc_performance.transport_discarded`; a discarded zero is sent again;
- a scheduler reset or restart while a commit is pending, the scheduler streams unavailable or owned by another producer, the own `_aux` channel unavailable, or no receipt within `dm.receiptTimeoutMs`. These count in `ddspc_performance.dm_write_failures` (an outage counts once), `safety.output_state` shows the latest as `write_failed_<reason>` until the next receipt, and the log notes "DDSPC DM write failed N time(s) (latest: ...); output continues" at most once per second.

### Slot mapping

A slot ordinal is the scheduler `cycle*steps+step`. A WFS frame owns the q slots after the newest slot copied when it arrived. Training takes the q slots copied since the previous decision, oldest first; a missed slot holds the retained copy. The standard model receives their mean, which equals its held command.

In assisted mode, slot p of a frame is calculated with:

- the frame's WFS measurement and the WFS/command histories, published before learning starts;
- the k ACC samples received since the frame's anchor (the newest ACC sample at the WFS decision), at most `acc_ratio − 1`. With an ACC not synchronized to the WFS, k need not equal p: `acc_lagging` and `acc_anomalies` count slots with fewer, or more, new samples than slots, and are then not health measures;
- the p commands already executed in the frame: the scheduler's copies, or the retained copy for a slot that was not copied.

A slot with no current frame (a late WFS frame) holds the last command (`fast_holds`). `ddspc_performance.slot_shifts` counts decisions whose interval contained other than q new copies; a steadily rising count means the WFS and DM clocks are not locked.

### Within-frame ripple

The DDSPC cost penalizes future WFS frames, which integrate q slots, so command variation within a frame is unobserved. In simulation the unmodified cost (`ripple=0`) added within-frame ripple that made assisted control worse than standard when the ACC carried no information. `ripple` penalizes each future frame's variation about its mean command, scaled like `regularization`; with `ripple=10` and no ACC information assisted stayed within 5–10% of standard, and with informative ACC it reduced the 8 kHz residual variance by 60–80%. Re-solving with fixed executed slots remains exact for this cost (test T5). Measure the WFS residual and, if available, an independent fast sensor when tuning it.

### Integrator

The original loPredCtrl wrote its command into the main loop's modal input, so cacao `mfilt` integrated it: `DM[k] = mult·DM[k-1] − gain·delta[k]`, clipped to ±limit. Its model was trained on those deltas, and the main loop kept integral action on tip/tilt, which tweeter-to-woofer offloading relies on. Without the integrator (`enabled=false`) loPredCtrlAcc applies the same DDSPC calculation as an absolute position, and `parameters.gain·y` is proportional.

`[integrator] enabled=true`, the default, restores the mfilt law in both modes, with the main loop's own parameters; `enabled=false` keeps absolute positions. The output still goes through syncopation to `dm.channelName`; only the value written changes. Each command `delta` (including `gain·y` and probes) updates `s = mult·s − gain·delta`, clipped to ±`limit`, and synco writes `s`:

- **Standard:** one update per WFS frame, held on the frame's four slots, exactly as mfilt. A late WFS frame holds `s`.
- **Assisted:** one update per DM slot on the DM worker with `gain/dm_slots`; `mult` and `limit` are unchanged. A slot with no current frame holds `s`.

For each DDSPC mode `i` (WFS index `modal_index0/1`), loPredCtrlAcc follows the main loop as it changes:

| Parameter | Source |
| --- | --- |
| `gain` | `<loop.device>.loop_gain.current × aol<N>_mgainfact[i]` |
| `mult` | `<loop.device>.loop_multcoeff.current × aol<N>_mmultfact[i]` |
| `limit` | `<loop.device>.loop_max_limit.current × aol<N>_mlimitfact[i]` |

`loop.device` defaults to `holoop` and `loop.number` (N) to `1`. The combined values are published as `loPredCtrlAcc.integrator.*` (`enabled`, `ready`, `gain0/1`, `mult0/1`, `limit0/1`). Prediction is refused, and `safety.output_state` reports `integrator_parameters`, until all six have been received with positive gain and limit for both modes, so the main loop's tip/tilt gain (`loop_gain × mgainfact` for modes 0 and 1) must be nonzero. After that each keeps its last value until it is updated.

Each fit is trained on the deltas recovered from the copied commands, `−(p − mult·p_previous)/gain` per frame (standard) or per slot with `gain/dm_slots` (assisted), so missed slots and clipping are seen as applied. The assisted executed-slot prefix uses the same deltas, and `ripple` then penalizes delta variation within a frame. The state is zero whenever prediction is off, so prediction always starts from the confirmed zero; a fault zero still removes the whole state in one slot. A change of the main loop's gain changes the plant a frozen fit was trained on; keep learning on, or retrain, after one.

```bash
getINDI 'loPredCtrlAcc.integrator.*' 'holoop.loop_gain.current' 'holoop.loop_multcoeff.current' 'holoop.loop_max_limit.current'
```

The feedback sign matches mfilt only if `dm.modeCube` planes 0 and 1 use mfilt's DM mode sign. It is a startup setting; the restart that changes it also discards the fits, which would not apply in the other form. The design is in `agents/plans/2026-11/ddspc-standard-integrator.md`.

### Missed and late frames

The WFS and ACC histories are the newest measurements received, in arrival order. A missed or late frame neither breaks nor fills them:

- **WFS:** each frame received is the next history entry, whatever source frames were missed or how late it arrived. Both fits stay ready and the output stays qualified; learning continues.
- **ACC:** the window is the newest `acc_ratio × accel_history` samples received (each numbered on arrival), at whatever rate they arrive. A late sample takes its place when it arrives; a sample the source never delivered (lost in transfer, skipped by acc000) simply does not occur; a frame with no new samples reuses the newest ones. `assisted_qualification.acc_window` reads `ok`.
- **ACC reader:** every in-range sample is used as it arrives on the RTC. The producer's (Pi's) timestamps and clock, sample intervals and lost publications are only counted (`adc.max_interval_us` and `adc.clock_tolerance_us` are accepted for existing configurations); the Pi and RTC clocks need not agree.
- **Upstream lock:** not needed. acc000 may run free (`synchro.timingSource = integrator`, the default); the lock is shown as `ddspc_control.accel_sync` and in `assisted_qualification`, and gates neither learning, probes nor prediction.

`parameters.max_fill` is still accepted and only sizes the copied-slot ring. The window fails only before it first fills (`history_filling`) or when the ACC stream stops for longer than `adc.max_age_us` (`stale_sample`).

### Holding through lost input

A condition that would otherwise stop prediction at once (`wfs_stale`, `wfs_invalid`, `command_invalid`, and in assisted mode `acc_window`: the ACC stream silent for longer than `adc.max_age_us`) is held for up to `parameters.hold_timeout` seconds (default 1): no slot is rendered, so synco keeps the DM at its last command, and prediction continues by itself when the condition clears. Only a failure lasting longer stops prediction, with the condition that started it in `safety.output_state`. A request made while the input has never qualified is still refused at once. `ddspc_performance.input_holds` counts the holds; `hold_timeout=0` stops at once, as before.

The upstream lock never holds, stops or pauses anything.

## Before installation (operator, on the RTC)

```bash
hostname   # the RTC
grep -n destinations /opt/MagAOX/config/dmtweeter-synco.conf   # must list dm01disp05
ls -l /milk/shm/dm01disp05_aux.im.shm                          # must exist with the tweeter geometry
lslocks | grep -E 'dm01disp05_aux' || echo 'no producer owns dm01disp05_aux'
cat "$CGROUPS1_CPUSET_MOUNTPOINT/spare28/tasks" "$CGROUPS1_CPUSET_MOUNTPOINT/spare29/tasks"   # both empty
```

If `dm01disp05` is not a scheduler destination, synco never copies `dm01disp05_aux` and nothing reaches the DM: add it to the scheduler's `[streams] destinations` (a separate scheduler configuration change) before starting loPredCtrlAcc.

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
getINDI 'acc000.fps.current' 'loPredCtrlAcc.rates.*'   # acc000's rate, and acc_per_wfs / acc_ratio detected from it
```

Wait for `safety.zero_confirmed=true`, the consumer's receipt for the initial own-channel zero. If it stays false, check the scheduler, the channel and `safety.output_state`. If a readiness flag stays false, `ddspc_control.standard_state` and `assisted_state` give the reason. Possible reasons are an absent WFS, sensor history, insufficient training or excitation, and numerical faults. A refused `predict` or `learn_std` request is answered at once with its unchanged state and Alert; `predict` stays Alert until a request is accepted, which `loPredCtrlAccGUI` shows as a warning.

Start the disturbance and record a 60 s baseline with prediction off:

```bash
setINDI -n 'acc000.dm_scale.target=<approved scale>'
setINDI -s 'acc000.dm_enable.toggle=On'
```

## Assisted qualification

`ddspc_control.assisted_state` reports the first failing condition. `assisted_qualification.blockers` lists all of them, for example `acc_window_history_filling,model_untrained_0_of_212_updates`; `none` means assisted mode is qualified. The other elements give the values behind them: `adc_state`, `acc_window` (the WFS thread's ACC window: `ok`, `history_filling`, `stale_sample`, ...) and `model` (updates/needed). `upstream_enabled`, `upstream_locked` (`mode_code` and `delay_lock`, or `enabled` and `valid` for `ddspcTiming`), `upstream_phase_error_us`, `upstream_read_latency_error_us` and `upstream_age_s` report acc000's synchronization for information only; it is never a blocker.

```bash
getINDI 'loPredCtrlAcc.assisted_qualification.*'
getINDI 'acc000.timingDiag.delay_lock' 'acc000.timingDiag.delay_phase_error_us' 'acc000.timingDiag.read_latency_error_us'
getINDI 'acc000.sim_diag.lock_phase_ok' 'acc000.sim_diag.lock_timing_ok' 'acc000.sim_diag.lateness_window_max_us' \
        'acc000.sim_diag.acq_timer_slack_ns' 'acc000.sim_diag.acq_sched_policy' 'acc000.sim_diag.acq_priority'
```

With accelSim, `delay_lock=0` while `delay_phase_error_us=0` means late acquisitions (`lock_timing_ok=0`); see the accelSim README. After installing the updated accelSim, `acq_timer_slack_ns` should read 1, and `read_latency_error_us` and `lateness_window_max_us` should fall well below 50.

### Assisted learning without synchronization

The assisted fit learns, and exploration probes run, whether or not acc000 reports a lock. `parameters.assisted_diagnostic` is accepted for existing configurations and has no effect, and there is no longer a reset required for a fit trained without lock. To evaluate a trained assisted fit, keep learning off and read `rmse_phase0..(acc_ratio−1)`. Design: `agents/plans/2026-11/ddspc-free-running-acc.md`.

## Train and test standard

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=On;accel_assisted=Off'
setINDI -n 'loPredCtrlAcc.gain.target=0.3'
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.safety.*'
```

The triples are loPredCtrl's original sequence: WFS frames, Gaussian standard deviation in modal command units, and regularization. Use the middle value that matches the deployed modal scale. `setINDI -s 'loPredCtrlAcc.learn_std.toggle=On'` installs `1000,0.05,100,1000,0.05,3,1000,0.05,0.1,1000,0,0.1` and enables learning and prediction together: loPredCtrl's three stages, then 1000 frames of learning without probes. During the 4,000-frame sequence (2 s), `prediction=warming` shows probe or feedback commands. When the last stage ends, `learn` and `learn_std` turn off by themselves and prediction continues with the frozen fit; the log notes "DDSPC standard learning sequence finished". To keep learning for X more seconds first, set `learn_std_extra` (default 0, also `parameters.learn_std_extra`); a change applies at once, including during a running sequence:

```bash
setINDI -n 'loPredCtrlAcc.learn_std_extra.target=3'
``` The final regularization (0.1) stays in use until `reset_model`. Learning turned on by hand (`learn`) is never stopped by an exploration sequence. Check that learning has stopped and the fit is ready, then measure:

```bash
getINDI 'loPredCtrlAcc.ddspc_control.*'   # learning=false, standard_ready=true
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
- In assisted mode: `predict_rate_hz` near 8000 (a calculation for every slot, not a replayed plan); while learning, `update_rate_hz` and `refresh_rate_hz` near 2000 (one QR-RLS update per WFS frame); `fast_holds`, `acc_lagging`, `acc_anomalies`, `step_mismatch` and `learn_overruns` near zero; `acc_phase0..(acc_ratio-1)` equal; `margin_us_min` positive; `learn_us_p99` well below 500.
- `acc_age_calc_us_*` and `acc_age_apply_us_*` give the age of the newest ACC sample at each calculation and at the slot's DM target time. Lower `dm.renderLeadUs` while `margin_us_min` stays positive with headroom; raise it if `transport_skipped` rises.
- `safety.output_state` to report `transport_acknowledged_not_physical_application`.
- The scheduler's accepted count for the `dm01disp05` index (`accepted<k>`, where k is its position in `[streams] destinations`, counting from 0) to rise by about 480,000, with `missing<k>` and `rejected<k>` near zero.

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

## Main-loop route

`ddspc_mode.main_loop` is the standard model delivered the way the original loPredCtrl delivers it. Standard and assisted are unchanged. Each WFS frame:
- the standard decision (`gain*y` plus the learned prediction, plus probes while exploring) replaces modes 0 and 1 of the main-loop modal output, and every other mode is copied;
- the main loop's mfilt integrates it with its own gain, multiplier and limit, at the WFS rate;
- nothing is written to `dm.channelName` (`dm01disp05_aux`), and tip/tilt is never zeroed;
- the model trains on the tip/tilt it published.

| State | Main-loop modes 0 and 1 | `ddspc_control.main_loop_tt` |
| --- | --- | --- |
| Prediction off | unchanged WFS tip/tilt | `passthrough` |
| Prediction on (`predict` or `learn_std`) | DDSPC delta | `predicted` |

`requested_mode` reads `main_loop`. The integrator's parameters are not needed (`integrator.ready` may be false). With no model yet, prediction on sends `gain*y`: the main loop's own control with its gain scaled by `gain`, so `gain=1` changes nothing and is a safe first check. A numerical fault publishes the measurement and stops prediction with a logged reason.

```bash
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # zero_confirmed=true
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=Off;main_loop=On'
setINDI -s 'loPredCtrlAcc.learn_std.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*'   # requested_mode=main_loop, main_loop_tt=predicted
```

Changing `ddspc_mode` while predicting in this route stops prediction ("DDSPC prediction stopped: DDSPC mode changed"), and tip/tilt passes through from the next frame. `predict` off or `release_tt` also return tip/tilt on the next frame; there is no own-channel zero to wait for. The decision now precedes publication, so each frame's output waits for the standard decision, including its RLS update while learning (13.5 µs mean on a development Mac, 1.4 µs frozen). Details: `agents/plans/2026-11/ddspc-main-loop-route.md`.

## mfilt copy

`ddspc_mode.mfilt_copy_8k` and `mfilt_copy_2k` move the main loop's tip/tilt control onto the DDSPC DM channel without DDSPC. They compute the same thing and differ only in how often it is written. Each WFS frame (2 kHz) while `predict` is on:
- `s = clamp(mult*s - gain*y, ±limit)` for modes 0 and 1, where `y` is the `aol1_modevalWFS` tip/tilt and `gain`, `mult`, `limit` are the main loop's own (`holoop` `loop_gain`/`loop_multcoeff`/`loop_max_limit` times `aol1_mgainfact`/`mmultfact`/`mlimitfact`, the `integrator` property). This is mfilt's law;
- `s` is converted through `modeCube` planes 0 and 1 and written to `dm.channelName`:
  - `mfilt_copy_8k`: on all four DM slots of the frame, so synco writes the channel at 8 kHz with the value changing at 2 kHz;
  - `mfilt_copy_2k`: once per frame, on the first slot synco copies after the decision (a late slot is retried on the next). synco holds it on the other three slots, so the DM sees the same command. Those three slots count in `ddspc_performance.transport_skipped` (6000 per second), so that counter is not a health measure in this mode;
- main-loop modes 0 and 1 are zero (`main_loop_tt=zeroed`), exactly as in standard mode, so mfilt's tip/tilt state holds and the copy starts from zero on top of it.

No model is used or trained: `gain`, `learn` and exploration have no effect, and `learn_std` is refused. `predict` is refused until the main loop's parameters are received (`integrator.ready=true`), even with `[integrator] enabled=false`. Turning `predict` off zeroes the channel and returns tip/tilt to the main loop once `zero_confirmed=true`, as in standard mode.

If the loop holds as it does with the main loop alone, the DM channel path (shapes and sign of `modeCube`, synco timing, `dm01disp05` routing) is right, and any instability in standard or assisted mode comes from DDSPC. If it does not, the path is the problem. A sign error shows as tip/tilt running away within a few frames; turn `predict` off.

```bash
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # zero_confirmed=true
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=Off;main_loop=Off;mfilt_copy_8k=Off;mfilt_copy_2k=On'
getINDI 'loPredCtrlAcc.integrator.*'   # ready=1
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*'   # requested_mode=mfilt_copy_2k, prediction=active, main_loop_tt=zeroed
```

Details: `agents/plans/2026-11/ddspc-mfilt-copy.md`.

## mfilt clone

`ddspc_mode.mfilt_clone_8k` and `mfilt_clone_2k` are the closest this channel gets to the main loop itself: the only intended difference is that tip/tilt reaches the DM through synco. While `predict` is on:
- at the start, the clone's state `s` is read from modes 0 and 1 of `aol1_modevalDM`, mfilt's output (published as `mfilt_clone.start_tt0/1`); `predict` is refused if that stream is not available;
- each WFS frame, `s = clamp(mult*s - gain*y, ±limit)` with the main loop's parameters, as mfilt;
- main-loop tip/tilt is zero, so mfilt keeps its last output on its own channel, decaying by `mult`. The clone writes `s` minus that held output to `dm.channelName`, so the DM total is `s` and the switch has no jump;
- `mfilt_clone_2k` writes only step 0, the slot synco copies with the main loop's write; a decision that is not ready by then waits a frame for the next step 0. `mfilt_clone_8k` writes every slot.

Check that step 0 is in sync with the main loop:

```bash
getINDI 'loPredCtrlAcc.mfilt_clone.*'
# step0_in_sync   step 0 carried the decision for the frame the main loop's write carries
# step0_missed    the clone was a frame behind the main loop; should stay at or near zero
# step0_margin_us_min/mean   render minus decision completion; lower dm.renderLeadUs if missed rises with small margins
```

Select it like the copy, with `predict` off and `zero_confirmed=true`:

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=Off;main_loop=Off;mfilt_copy_8k=Off;mfilt_copy_2k=Off;mfilt_clone_8k=Off;mfilt_clone_2k=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*'   # requested_mode=mfilt_clone_2k, main_loop_tt=zeroed
```

The held output is modelled from the start value, not re-read. If `aol1_modevalDMf` feedback rewrites mfilt's state, the DM total differs from `s` by that. Details: `agents/plans/2026-11/ddspc-mfilt-clone.md`.

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

- a stale or invalid WFS stream, or in assisted mode an ACC stream silent for longer than `adc.max_age_us`, lasting longer than `parameters.hold_timeout` (see "Holding through lost input");
- a numerical fault;
- a non-finite command, or an exception while rendering or after a copy.

Failed DM writes are not on this list (see "DM channel, worker and timing"). A wrong `_aux` geometry is a configuration error and still stops the output.

`safety.output_state` gives the reason, and keeps it in parentheses after the zero is acknowledged, for example `fault_zero_acknowledged (input_or_model_qualification_lost: wfs_stale)`. Each fault is also logged as "DDSPC output fault, prediction stopped". A lost qualification names the first failed condition: `wfs_invalid`, `integrator_parameters` (the main loop's parameters are missing), `command_invalid`, `wfs_stale` (no WFS frame within `input_timeout`), or `acc_window` with the ACC window reason. After confirmed removal, prediction can be re-enabled, or the models reset and retrained (`setINDI -s 'loPredCtrlAcc.reset_model.request=On'`). A normal exit also attempts an acknowledged own-channel removal.

Every stop of prediction and every refused request is also logged as a note, with its reason: `DDSPC prediction stopped: <reason>` (turned off by request, Zero DDSPC DM requested, or tip/tilt returned to the main loop), `DDSPC prediction request refused: <reason>` and `DDSPC standard learning sequence refused: <reason>`. Faults keep their warning entry. To list them all:

```bash
logdump --nfiles=3 loPredCtrlAcc | grep 'DDSPC '
```

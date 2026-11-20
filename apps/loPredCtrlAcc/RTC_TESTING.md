# loPredCtrlAcc laboratory operation

The normal build runs closed-loop DDSPC on the existing WFS monitor (`aol1_modevalWFS`, 2 kHz). It publishes a full calibrated surface through the syncopation producer to its own auxiliary channel, `dm.channelName`.

- **Standard mode** follows loPredCtrl. It is single-rate: one command per WFS frame, held on that frame's four DM slots. It does not use the accelerometers and runs with `adc.enabled=false`.
- **Assisted mode** adds the upstream-synchronized 8 kHz X/Y samples available at each WFS decision. It plans four distinct slot commands per frame (`ShadowConfig::m_assistedCommandSlots = 4`), one per 8 kHz DM slot.

Both models learn only from genuine WFS measurements and from the commands the scheduler actually copied from this channel.

## DM channel, worker and timing

| Setting | Value | Reason |
| --- | --- | --- |
| `dm.channelName` | `dm01disp09_aux` | One producer per scheduler destination. The `<aux>.producer` lock is exclusive, and synco copies the whole image. `acc000` owns `dm01disp10_aux`, so the disturbance and the correction must use different destinations; dmcomb sums them. The code accepts any `*_aux` name. |
| `dm.threadPrio` | `20` | Dedicated producer worker started by `MagAOXApp::threadStart`, as in accelSim and dmSpeckle. It runs below the scheduler's priority 30. |
| `dm.cpuset` | `spare29` | Dedicated development core (CPU 29). acc000 uses `spare30` and `spare31`. |
| `shmimMonitor.cpuset` | `spare28` | Runs the WFS decision and RLS/QR learning, about 140 µs per frame while learning on a development Mac. |
| `syncopation.writeLeadUs` | `125` | The worker renders a slot as soon as the slot is eligible. A 125 µs lead commits it at once, which maximizes the margin before the cutoff. |

The worker polls every 10 µs and checks the next slot immediately after a commit. A slot that cannot be committed by its cutoff is counted in `ddspc_performance.transport_skipped`; synco then keeps the last copied contribution. Such a slot no longer latches output off.

### Slot mapping

A slot ordinal is the scheduler `cycle*steps+step`. Each decision indexes its plan from the newest slot copied when its WFS frame arrived. Training takes the four slots copied since the previous decision, oldest first. A missed slot holds the retained copy. The assisted model receives the four values as a lift; the standard model receives their mean, which equals the held command in standard mode.

`ddspc_performance.slot_shifts` counts decisions whose interval contained other than four new copies. Occasional shifts can occur when a WFS arrival coincides with a copy. A steadily rising count means the WFS and DM clocks are not locked 4:1.

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

Wait for `safety.zero_confirmed=true`, the consumer's receipt for the initial own-channel zero. If it stays false, check the scheduler, the channel and `safety.output_state`. If a readiness flag stays false, `ddspc_control.standard_state` and `assisted_state` give the reason. Possible reasons are an absent WFS, the upstream lock, sensor history, insufficient training or excitation, and numerical faults.

Start the disturbance and record a 60 s baseline with prediction off:

```bash
setINDI -n 'acc000.dm_scale.target=<approved scale>'
setINDI -s 'acc000.dm_enable.toggle=On'
```

## Train and test standard

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=On;accel_assisted=Off'
setINDI -n 'loPredCtrlAcc.gain.target=0.3'
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.safety.*'
```

The triples are loPredCtrl's original sequence: WFS frames, Gaussian standard deviation in modal command units, and regularization. Use the middle value that matches the deployed modal scale. `setINDI -s 'loPredCtrlAcc.learn_std.toggle=On'` installs the same triples and enables learning and prediction together. During the 3,000-frame sequence (1.5 s), `prediction=warming` shows probe or feedback commands. When `standard_ready=true`, freeze the fit and measure:

```bash
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
# measure 60 s with the standard correction applied
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # wait for zero_confirmed=true before changing mode
```

## Train and test assisted

Train assisted with assisted selected. Its probes are then drawn independently for each of the four slots, which identifies the per-slot response. Learning updates both fits. The standard fit receives the frame mean of the same commands, which is valid single-rate data, so frozen A/B comparisons remain fair.

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

For an A/B comparison, keep both fits frozen and the disturbance unchanged, and alternate modes. Switch modes only after `zero_confirmed=true`. No reset is needed. `matched_standard_rmse` and `matched_assisted_rmse` score identical forecast origins, targets, horizons and channels while `matched_targets` advances. Optical correction is judged from the WFS residual with prediction off and on.

## 60 s publication check

Take `getINDI 'loPredCtrlAcc.ddspc_performance.*'` and `getINDI 'dmtweeter-synco.production.*'` 60 s apart with prediction active. Expect:

- `transport_published` and `transport_acks` to rise by about 480,000.
- `transport_skipped` and `slot_shifts` to stay near zero.
- `safety.output_state` to report `transport_acknowledged_not_physical_application`.
- The scheduler's accepted count for the `dm01disp09` index (the third destination, `accepted2`) to rise by about 480,000, with `missing2` and `rejected2` near zero.

## Remove the contribution

```bash
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'   # zero_confirmed=true
setINDI -s 'acc000.dm_enable.toggle=Off'
xctrl stop loPredCtrlAcc
```

Prediction turns off and own-channel zero is requested when any of these occur:

- a discarded receipt;
- a stale WFS stream or a WFS counter gap, which breaks the model history;
- invalid selected sensor data;
- a numerical fault.

`safety.output_state` gives the reason. After confirmed removal, prediction can be re-enabled, or the models reset and retrained (`setINDI -s 'loPredCtrlAcc.reset_model.request=On'`). A normal exit also attempts an acknowledged own-channel removal.

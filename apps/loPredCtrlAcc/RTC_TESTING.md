# loPredCtrlAcc laboratory operation

The normal build runs closed-loop DDSPC. Both modes use the existing WFS monitor and publish a full calibrated surface to `dm10disp10_aux` through the syncopation producer. Start with standard mode. It needs the WFS, the DM calibration, and the DM producer/consumer. It can run with `adc.enabled=false`.

## Build and install on the RTC

Run these commands as the operator in the existing MagAO-X build environment. Stage the configuration before stopping the running application. Compare the staged settings with the active WFS and DM calibration, and carry over the ADC settings that are already delivering valid data. The two `diff` commands print differences. The editor opens the staged copy.

```bash
cd ~/source/MagAOX-DDSPC/apps/loPredCtrlAcc
make -C ../.. flatlogs/bin/flatlogcodes
make -j1 DDSPC_OPERATIONAL_BUILD=yes all
rtc_config=$(mktemp --suffix=.conf)
cp loPredCtrlAcc.conf "$rtc_config"
diff -u /opt/MagAOX/config/loPredCtrl.conf "$rtc_config"
diff -u /opt/MagAOX/config/loPredCtrlAcc.conf "$rtc_config"
"${EDITOR:-vi}" "$rtc_config"
./loPredCtrlAcc -n loPredCtrlAcc -c "$rtc_config" --config.validate
xctrl stop loPredCtrlAcc
make -j1 DDSPC_OPERATIONAL_BUILD=yes install
sudo install -m 0644 --backup=numbered "$rtc_config" "$(readlink -f /opt/MagAOX/config/loPredCtrlAcc.conf)"
/opt/MagAOX/bin/loPredCtrlAcc -n loPredCtrlAcc --config.validate
```

The complete RTC configuration is [loPredCtrlAcc.conf](loPredCtrlAcc.conf). Use the current deployed `loPredCtrl.conf` for the WFS stream and controller settings if they differ. For the already-running application the xctrl/INDI registrations can be reused.

| Setting | Supplied value and source |
| --- | --- |
| WFS input | `aol1_modevalWFS`, FLOAT, entries 0 and 1, from the repository's native input configuration |
| Standard fit | Two modes, history 15, future 3, QR-RLS, gamma 1, prior 1e-5, regularization 100, additive feedback gain 0.3, from [the recorded operational parameters](tests/operational_parameters.ini) |
| Output | `dm10disp10_aux`, existing `dmtweeter_synco_timing`, producer lead 50 microseconds |
| DM basis | The configuration names the tweeter-vispyr `CMmodesDM.00.fits` cube. The application uses its first two planes at their stored scale and in native pixel order. It computes the inverse projection to read back its own modal commands. An optional matching mask zeros pixels outside the support. |
| Assisted input | The shipped profile uses the repository's `acc000` simulator encoding (2048 zero, 0.0083799 V/count, 1.02 V/(m/s²), positive polarities) and `acc000.timingDiag`. The upstream producer must already synchronize to the WFS at 4:1. |
| Physical ADC | If using AD7689, retain the **working RTC ADC calibration** from the old controller config: stream, profile label, code limits, per-axis zero, polarity, volts/count and sensitivity. Set `adc.sync_device` to its actual INDI name and `adc.sync_property=ddspcTiming`. The simulator encoding does not establish these hardware values. |

The instrument-specific optical question is whether the active WFS indices and signed coefficient units correspond to those two synthesis planes on this routing. Check the deployed modal-to-DM calibration. If the basis requires a permutation, a scale, a sign change, or a different mask, use the deployed planes or supply the existing full mapping with `dm.calibration` in [DDSPC_MODAL_V1 format](SYNCO_NATIVE.md). The text mapping supplies its own WFS indices. A dimension or rank check cannot establish optical sign. Keep the original loPredCtrl correction out of the experimental contribution unless deliberately testing the combined loop.

## Train and test standard first

```bash
xctrl start loPredCtrlAcc
getINDI 'loPredCtrlAcc.safety.*' 'loPredCtrlAcc.ddspc_control.*'
```

Wait for `safety.zero_confirmed=true`. This is the existing consumer's receipt for the initial own-channel zero. If it stays false, check the active scheduler, ownership and auxiliary routing. Prediction starts off. If a readiness flag stays false, read `ddspc_control.standard_state` or `assisted_state`. These distinguish absent WFS, upstream lock, sensor history, insufficient training or excitation, and numerical faults.

```bash
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=On;accel_assisted=Off'
setINDI -s 'loPredCtrlAcc.reset_model.request=On'
setINDI -n 'loPredCtrlAcc.gain.target=0.3'
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.safety.*'
```

The three triples are the original loPredCtrl standard sequence, in WFS frames, Gaussian standard deviation in **modal command units**, and regularization. Use that sequence only with its corresponding deployed modal scale. For a different lab probe amplitude, change the middle value of each triple. The original shortcut `setINDI -s 'loPredCtrlAcc.learn_std.toggle=On'` installs those same triples and enables learning and prediction together. Learning alone can advance the update counters against an acknowledged zero but needs varying commands to identify a DM response. With nonzero gain, the two toggles alone can start feedback and identification from naturally varying measurements. The explicit sequence also excites the plant when the disturbance provides little variation. During the initial fit, `prediction=warming` identifies feedback or probe commands. Once the fit is ready, `awaiting_copy` changes to `active` when the consumer has acknowledged an active-mode payload.

After the 3,000-frame sequence, confirm `standard_updates` increases and `standard_ready=true`, then freeze the fit and observe the closed-loop WFS residual. While prediction is active, check that both `transport_published` and `transport_acks` advance and that `safety.output_state` reports a copied request. Compare the measured WFS residual with prediction off and on under the same disturbance to verify the sign and magnitude of correction.

```bash
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
getINDI 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.ddspc_control.*'
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'
```

Wait for `zero_confirmed=true` before changing modes. `gain` adds current-residual feedback to the learned command. Set `predict.toggle=Off` to remove the whole contribution.

## Train and test assistance

With synchronized ACC present during the preceding training, both fits have already trained on the same WFS and command records. Check `adc_values.valid=1`, the upstream lock, `assisted_updates` and `assisted_ready`. For an A/B comparison, keep the existing fits frozen and use the same disturbance and gain.

```bash
getINDI 'loPredCtrlAcc.adc_values.*' 'loPredCtrlAcc.ddspc_control.*' 'loPredCtrlAcc.ddspc_performance.*'
setINDI -s 'loPredCtrlAcc.ddspc_mode.standard=Off;accel_assisted=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.safety.*' 'loPredCtrlAcc.ddspc_control.*'
```

If ACC became available **after** standard training, select assisted mode after confirmed removal and use these commands to train before the frozen test. No model reset is needed.

```bash
setINDI -x 'loPredCtrlAcc.exploration_sequence.target=1000,0.1,100,1000,0.1,3,1000,0.1,0.1'
setINDI -s 'loPredCtrlAcc.learn.toggle=On'
setINDI -s 'loPredCtrlAcc.predict.toggle=On'
getINDI 'loPredCtrlAcc.ddspc_performance.*' 'loPredCtrlAcc.ddspc_control.*'
# Once the sequence has run and the assisted fit is ready:
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
```

Both models make a decision on each 2 kHz WFS publication. The assisted state includes the most recent contiguous 20 individual X/Y samples available at that decision (five WFS periods at 8 kHz). The application consumes the upstream lock and preserves sample order. It does not estimate phase, fit counter offsets, interpolate, or synthesize missing samples. Both modes hold the WFS-rate result for the producer's advertised DM slots. Compare `matched_standard_rmse` and `matched_assisted_rmse` only while `matched_targets` advances. These score identical forecast origins, targets, horizons and channels. Compare residual telemetry across actual loop closures to evaluate correction.

## Remove the contribution

```bash
setINDI -s 'loPredCtrlAcc.predict.toggle=Off'
setINDI -s 'loPredCtrlAcc.learn.toggle=Off'
getINDI 'loPredCtrlAcc.safety.*'
# After zero_confirmed=true:
xctrl stop loPredCtrlAcc
```

A missing or discarded receipt, stale WFS, invalid selected sensor data or numerical failure can interrupt output. Inspect `safety.output_state`. A missing zero receipt leaves removal unconfirmed. After confirmed removal, the operator can select standard or reset and retrain. A normal exit also attempts acknowledged own-channel removal.

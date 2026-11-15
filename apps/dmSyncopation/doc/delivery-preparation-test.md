# Measure controller delivery and preparation

This test separates two parts of the command path that the pipeline shadow has
represented with fixed allowances. It measures the existing genuine 2 kHz path
while auxiliary commands remain hypothetical. The new diagnostics do not change
DM submissions, automatically tune simulation settings, credit a minimum future
main preparation time, or change the genuine-main combined-observer anchor.
The [engineering record](../../../agents/plans/2026-10/dm-syncopation-release-readiness-shadow.md)
contains the optimized preparation captures and the first main-wait trace run.

```text
dm01disp03 publication
     |
     | dmcomb reacts and combines
     v
dm01disp publication
     |
     | deliveryLower ... deliveryUpper
     v
BMC semaphore receipt
     |
     | preparation
     v
SDK dispatch
     |
     | sdkDispatch (handoff / possible queueing)
     v
native SDK entry -> SDK service -> worker ready
```

`preparation` uses the paired controller receipt and SDK-dispatch timestamps
already captured for each command. It covers callback/admission work, conversion,
source checks and shape publication before handoff. It adds no hot-path clock
read. `sdkDispatch` covers handoff through actual native entry, including any
occupied-owner wait. `receiptSdk` covers both intervals and remains useful as a
conservative pre-SDK prefix; none of these is just the conversion loop.

Delivery uses the combined image's producer `writetime` and the controller's own
monotonic semaphore-receipt timestamp. An explicit producer-clock selection and
bracketed clock pair bound their relationship. `deliveryLower` and
`deliveryUpper` are the resulting per-command publication-to-receipt bounds;
`deliveryBracket` records the clock-pair uncertainty. This excludes synco's
observer wake and cycle-lock delays.

## Configure and rebuild

The operator performs these steps on the instrument. Keep the recorded camera,
loop, dmcomb mode/delay, CPU/IRQ placement and workload unchanged, including modal
gains. Retain RTC latency tuning On, the BMC SDK worker/full pipeline, and the
existing early/full completion streams. Synco stays in shadow mode with
operational `enable.toggle=Off`.

1. Rebuild/install **bmcCtrl and logdump**. The instrument's `teldump` is the alias
   `logdump --dir=/opt/MagAOX/telem --ext=.bintel`, so there is no separate
   `teldump` executable to rebuild.
2. Update the existing `[dm]` section of **dmtweeter.conf**, the BMC controller's
   instance configuration, with:

   ```ini
   [dm]
   deliveryClock=isio
   deliveryBracketUs=1
   ```

   `isio` explicitly selects the controller build's `CLOCK_ISIO`, which the
   inspected dmcomb producer uses for image publication. It is not automatic
   clock detection. Other explicit selections are `realtime` and `tai`; the
   selected clock must match the producer. The default `none` leaves delivery
   diagnostics disabled and adds no delivery clock reads. The configured
   bracket limit is one microsecond; excessively wide or invalid samples are
   retained as quality failures rather than usable delivery statistics.
3. Restart the BMC controller through the normal operator procedure, then restore
   the recorded loop state. Keep its `[sdk]` worker/pipeline settings unchanged.
   No camera rebuild or reconfiguration is required for these diagnostics.
4. Merge the updated additive
   [`dmtweeter_synco_pipeline_shadow.conf`](../dmtweeter_synco_pipeline_shadow.conf)
   into the existing **dmtweeter-synco.conf**, preserving its stream mappings and
   observer/core settings, then restart synco. The only changed model duration
   is the explicit provisional `simulation.prepareUs=25`; its compiled default
   remains 35 us. Keep delivery **20 us**, SDK service **120 us**, post service
   **150 us** and virtual readers **60 us**. Keep all guards/early-main allowances
   and zero minimum next-main preparation credit.

The 25 us shadow value is 2.658 us above the largest measured 22.342 us
receipt-to-SDK-entry maximum across optimized captures `212117Z` and `234448Z`.
That margin is provisional, and the measurement includes worker handoff. It does
not qualify an auxiliary path or guarantee an upper bound at 6 kHz.

## Verify the readbacks and capture

After at least ten seconds of stable cadence, the operator can run:

```bash
timeout -k 2s 8s getINDI dmtweeter.dm_timing_delivery
timeout -k 2s 8s getINDI dmtweeter.dm_timing_state
timeout -k 2s 8s getINDI dmtweeter.dm_timing_stages
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_pipeline
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_main_wait_trace
timeout -k 2s 8s getINDI sysMonRTC.set_latency.toggle
```

`dm_timing_delivery` reports `enabled`, resolved `clock`, `bracketUs`, `attempted`,
`qualified`, `unavailable`, `invalid`, `wideBracket`, `clockDiscontinuity`,
`futurePublication` and `identityRejected`. Numeric clock 0 means disabled,
1 means `CLOCK_REALTIME` and 2 means `CLOCK_TAI`; the text
`dm_timing_state.deliveryClock` also identifies the resolved selection. Confirm
that qualified counts advance, the bracket limit is the selected value, and
rejection/discontinuity counts are recorded with the usable coverage.

The new `dm_timing_stages` metric prefixes are `preparation`, `deliveryLower`,
`deliveryUpper` and `deliveryBracket`; each has the usual `mean`, `jitter`, `min`,
`max`, `count` and `span` suffixes. Durations are in **seconds**. These metrics and
delivery quality also appear in `[dmtimes]` telemetry. Absent/disabled or rejected
delivery is unavailable, rather than a measured zero. A narrow clock bracket
limits sampling uncertainty; it does not independently establish that the chosen
producer clock is correct. Qualification requires the full clock bracket to
follow the observed publication: `deliveryUpper < deliveryBracket` is counted
as `futurePublication`, even if the nominal upper value is positive. Such a
sample is rejected rather than clipped to zero.

Adjacent contiguous samples also compare their selected-clock-minus-monotonic
offset intervals, with a 100 ns capture/quantization tolerance. A detected step
is counted as `clockDiscontinuity` and rejected; a fresh baseline can then be
established. This check does not prove clock continuity across missing samples,
a producer restart, or the entire interval before the first clock pair. Retain
these quality changes when deciding whether a captured window can support a
latency bound.

Repeat the existing [simultaneous capture block](dmSyncopation.md#capture-a-simulation-run).
It records `dmtweeter-synco`, `dmtweeter` and `camwfs` plus bounded before/after
snapshots. Start with its 120-second duration; changing the collector duration
from `120s` to `600s` extends the same comparison to ten minutes. Keep complete
telemetry lines and source filenames, including the delivery quality records.
Save the live dmcomb trigger mode/delay and confirm modal gains and workload
with the capture.

## Interpret the measurements

- Compare paired `preparation` statistics with `receiptSdk` and `sdkDispatch` to
  separate preparation from SDK-worker handoff/queueing. Do not subtract the
  separate metrics' extrema to claim a per-command maximum or minimum.
- Use `deliveryUpper` and its coverage to assess the combined-publication to
  actual controller-receipt part of delivery. The clock-bracket width and all
  invalid/rejected samples remain part of the result. Counter/publication
  progression and source revalidation reject detected identity changes, but the
  metadata is still best-effort: it does not prove an atomic association between
  a particular semaphore token, frame pixels and producer timestamp. A newer
  publication observed only after the receipt clock bracket is rejected.
- Existing `combinerPrimaryPublication` measures the genuine primary publication
  through dmcomb's combined publication when the named-primary trace is usable.
  Together with the new controller delivery measurement, it covers the genuine
  primary-to-controller path. The two rolling populations are not necessarily
  command-paired: adding their means does not establish an exact end-to-end
  event, and summing compatible maxima is only a conservative aggregate bound.
- No auxiliary trigger is issued in this shadow. A measured genuine-primary path
  cannot yet qualify the future auxiliary trigger route. The current model still
  applies delivery after its genuine-main combined-observer anchor; these new
  measurements do not silently retime that anchor.
- Review collection quality and `main_wait` provenance before interpreting model
  stalls. A measured minimum preparation interval is not a guaranteed minimum
  for a future main, so this test does not credit it against SDK clearance.

With preparation 25 us and the other selected assumptions, the final auxiliary
trigger at 333.333 us reaches modeled SDK readiness at
**333.333 + 20 + 25 + 120 = 498.333 us**. The protected next-main boundary is still
**475 us**, leaving a nominal **23.333 us** shortfall. Even zero delivery would
leave a 3.333 us shortfall with preparation 25 us and SDK service 120 us. This
measurement/preset update does not make the final slot feasible or qualify
physical 6 kHz operation; the result supplies evidence for the next explicit
choice of bounds.

## Post-worker scheduling comparison on spare31

The `171940Z` baseline measured a 2.698 ms post/finish tail and three new full-
credit overloads while the post worker used ordinary priority and a broad CPU
mask. The operator confirmed **spare31 is free** for this comparison. The RTC
cpuset script assigns it to CPU31 with SMT sibling CPU95 offline.

Add these two keys to the existing `[sdk]` section of **dmtweeter.conf**:

```ini
[sdk]
pipelineThreadPrio=30
pipelineCpuset=spare31
```

These keys configure the ordered **post worker**, named `pipeline`; they do not
change the SDK worker's `threadPrio=55` or CPU56 placement. The historical
comparison used `spare56`; the updated RTC map names CPU56 `dmtweeter-sdk`.
Follow the [worker map migration](../../bmcCtrl/doc/sdk-worker.md#configuration)
before selecting the new name. Keep the early-release observer on CPU57
(`dmtweeter-synco-release`, historically `spare57`), all shadow bounds and credit
capacity unchanged, and camera at 2 kHz with synchronization On. The current
binary already supports these settings; this comparison needs a controller
restart, not another rebuild.

Restart dmtweeter through the normal operator procedure and restore the recorded
loop state. Check the actual post-worker readbacks:

```bash
timeout -k 2s 8s getINDI dmtweeter.sdk_pipeline
timeout -k 2s 8s getINDI dmtweeter.th-pipeline
```

Expect `sdk_pipeline.policy=1` (FIFO), `priority=30` and `allowedCpus=1`.
Startup also verifies the configured cpuset and scheduling settings. After at
least ten seconds of stable camera/main cadence and resource reacquisition,
repeat the same simultaneous capture. Compare capture-local changes in
`overloaded`, `rejected` and `coalesced`, post/saturation/publication tails, and
model release/miss coverage. Preserve the before/after snapshots: restart changes
counter baselines, so historical totals from `171940Z` are not new losses.

This changes post-worker policy and placement together. Improvement would support
using that configuration; it would not separately identify the contribution of
CPU placement versus FIFO scheduling. It also does not by itself qualify a shorter
post bound or physical 6 kHz operation.

The next experiment is the [coordinated zero-auxiliary test](zero-aux-rate-test.md):
retain the running 2 kHz main loop and add one or two scheduled zero auxiliary
commands per main cycle for 4/6 kHz effective total cadence. Its explicit finite
test mode uses actual reader clearance and measures main impact; the normal
shadow admission rules remain unchanged.

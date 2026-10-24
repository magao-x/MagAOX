# Isolated camera publication and acquWFS timing test

This test compares camera acquisition intervals, camera publication intervals and
`acquWFS-1` iteration-start intervals while the downstream AO stages are stopped.
The only running AO data-producing stages are `camwfs` and `acquWFS-1`;
`dmSyncopation` runs separately as a metadata observer. The operator performs all
instrument actions below.

The [camera-acquWFS.conf profile](../camera-acquWFS.conf) selects
`diagnostics.only=true`. This mode prevents enabling the scheduler, skips DM
resource attachments and schedule-stream creation, and automatically obtains an
available primary reader semaphore. It reads `aol1_imWFS2` and `camwfs` without
changing their pixels, counters, timestamps or existing reader assignments.
The combined-output observer and camera INDI FPS subscription are disabled.

The profile retains the two-second diagnostic window, 20000-event capacity,
20 us excursion threshold, and enabled camera/pipeline diagnostics. Its primary
observer runs at ordinary scheduling priority without the scheduler's default
`spare61` CPU placement. Camera observer scheduling also retains its ordinary
priority default. Keep the camera and `acquWFS-1` configuration, CPU placement,
priorities and acquisition rate unchanged during the comparison.

## Prepare the isolated producer chain

1. Build and install the matching `dmSyncopation` and `logdump`. The application
   must include diagnostic-only mode; the reader must understand camera
   publication timing. Record their Git hashes with the test results.
2. Open the AO loop using the normal operator controls, then stop the normal
   `dmtweeter-synco` instance. Record which downstream RUN processes were active
   so they can be restored afterward.
3. Use the existing CACAO FPS interface. The usual terminal setup is:

   ```bash
   cd /opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir
   cacao-fpsctrl-TUI
   ```

4. Select each of these rows and use **Ctrl+r** to stop its RUN process, in
   this order:

   - `mvalC2dm-1`
   - `mfilt-1`
   - `wfs2cmodeval-1`

   Leave their CONF processes running. Leave `camwfs` and `acquWFS-1` running;
   do not change acquisition, trigger, calibration or processing settings.
5. In `milk-procCTRL`, confirm the downstream RUN processes have stopped and
   `acquWFS-1` continues running. In `milk-streamCTRL`, confirm `camwfs` and
   `aol1_imWFS2` counters continue progressing at the camera cadence. The
   `aol1_imWFS2` producer trace should identify `acquWFS-1` with `camwfs` as its
   trigger input. Record its current PID and the camera input inode.

Stopping RUN processes through FPS leaves their configuration available for
restart. This procedure does not require a calibration reload or CONF restart.

## Start and record the observer

From the root of the MagAOX source checkout on the instrument, start a separate
instance in its own terminal or tmux pane:

```bash
/opt/MagAOX/bin/dmSyncopation -n wfs-acqu-timing -c "$PWD/apps/dmSyncopation/camera-acquWFS.conf"
```

The instance name selects separate logs and telemetry. The `-c` option loads the
explicit profile after any instance configuration. Use an otherwise unused
instance name, and check the startup configuration log for
`diagnostics.only=true`, primary stream `aol1_imWFS2`, and
`diagnostics.combiner=false`. Do not run a second observer with the same name.
The normal enable switch remains Off and enable requests are rejected in this
mode.

After startup, allow several diagnostic windows to fill, then record at least
two minutes with the current `teldump` alias and updated `logdump`:

```bash
teldump -f wfs-acqu-timing | tee wfs-acqu-timing.txt
```

Use Ctrl+C to stop the reader after the capture. Retain the text and binary
telemetry together with the capture times, camera rate, producer PID/inode and
the downstream RUN states. The normal instrument INDI server may not route a
new instance automatically; telemetry collection does not depend on reading its
properties through that server.

While diagnostics are enabled, ordinary application updates record telemetry
about once per second independently of the telemeter's ten-second forced-record
interval. Each record describes a two-second retained window, so normal records
overlap. Check actual record times, window spans and ages to establish coverage;
an application stall can leave a gap. Overlapping records share observations.
Use separated windows when totaling excursion counts.

## Validate collection before interpreting excursions

Check these parts of the `[dmsynco]` records:

- Camera acquisition and publication collectors are enabled and have populated
  statistics. `cameraWriteInterval` must appear after collection fills. If the
  publication table remains unavailable, check the application/reader versions
  and camera attachment before drawing conclusions.
- `pipeline_stages=1` identifies the expected one-stage trace. Its
  `pipeline_stage` row zero must have the recorded `acquWFS-1` PID and camera
  trigger inode. Here `pipeline_stage0` means the observed acquisition/preparation
  process, not the scheduler's passive step zero.
- Camera, primary and pipeline quality counters show no new dropped events,
  counter gaps, duplicates, rejected/unstable metadata or invalid traces in the
  comparison windows. Stale age, a partial span or a changing producer identity
  also limits coverage. Initial startup/reconnection counters should be
  distinguished from changes during the capture.
- At 2 kHz, a fully covered two-second interval population contains about 4000
  samples. Counts can differ slightly at window boundaries. A large discrepancy
  requires checking coverage before interpreting a quiet min/max.

The camera and primary monitors are independent. Their observation times and
window boundaries need not match exactly, so compare overlapping time ranges
and retain the quality fields alongside the timing statistics.

## Interpret the three timestamp series

| Text population | Interval being measured | Interpretation |
| --- | --- | --- |
| `cameraAcquisitionInterval` | Consecutive `camwfs.atime` values | EDT PDV SDK acquisition timestamp cadence. |
| `cameraWriteInterval` | Consecutive `camwfs.writetime` values | Camera publication timestamp cadence, after loading the image into shared memory. |
| `pipelineSourceInterval` | Consecutive oldest represented trace-start timestamps | `acquWFS-1` iteration-start cadence after its input-trigger wait. |
| `cameraWriteReceiptInterval` | Consecutive camera observer receipt timestamps | Observer scheduling/receipt cadence; useful for collection diagnosis. |

All durations in these populations are seconds. At 2 kHz the nominal interval
is 0.0005 s. Inspect minima and maxima as well as jitter: rare excursions can
remain visible in the extrema while RMS is small. `cameraWrite_pairs` reports
late/early ordering for camera publication intervals around their own retained
window median and the configured 20 us threshold.

The first-stage trace start is a **wake proxy**. It marks the MILK iteration's
processing start after the input wait; it is not a timestamp taken exactly when
the semaphore wait returns, and it is not a camera interrupt timestamp.

- If acquisition is steady but publication has large tails, investigate the
  camera path between SDK acquisition timestamps and shared-memory publication.
- If camera publication is steady but `acquWFS-1` start intervals have large
  tails, investigate publication-to-processing-start scheduling and trigger
  handling.
- If publication and first-stage intervals both have similar tails, variation
  is already present at camera publication. Independent marginal statistics do
  not establish that particular extrema occurred on the same frame.

These are comparisons of independent interval distributions. They do not
measure absolute camera-to-`acquWFS-1` latency or justify subtracting timestamps
from different clock domains. The observer's receipt intervals include its own
scheduling and cannot substitute for camera publication timestamps.

## Restore normal operation

1. Stop the standalone `wfs-acqu-timing` observer with Ctrl+C and let it exit.
2. Keep the AO loop open. In `cacao-fpsctrl-TUI`, use **uppercase R** to restart
   only the RUN rows that were active before the test, in dataflow order:
   `wfs2cmodeval-1`, then `mfilt-1`, then `mvalC2dm-1`.
3. Confirm normal stream progression and the expected process configuration
   before closing the AO loop through the usual operator procedure.
4. Restore the normal synco instance only with its normal configuration and the
   desired disabled/enabled state. The isolated test profile belongs to the
   standalone observer.

See the [main user guide](dmSyncopation.md) for diagnostic populations,
qualification rules and ordinary scheduler configuration. Median-grid
scheduler revisions remain separate from this measurement test.

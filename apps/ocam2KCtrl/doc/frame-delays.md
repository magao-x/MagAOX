# OCAM frame-delay diagnostics

An OCAM camera can fall behind while its raw frame numbers remain consecutive.
This monitor compares the cadence at EDT DMA completion (`atime`), SDK return
(`arrival`), and the timestamp immediately before shared-memory publication
(`write`). It uses the camera's reported rate or the external sync frequency.

Enable the monitor in the camera configuration and restart the application.
Framegrabber paired timing must be enabled (`latencyTime` and `latencySize`
positive). Detailed stage timing is optional.

```ini
[frameDelay]
enabled=true
historySec=900
thresholdFrames=1
dumpDir=frame-delays
```

An interval latches when its excess above the nominal period reaches
`thresholdFrames` periods. The integrated check looks at the most recent
`round(FPS)` consecutive intervals, with a minimum of one. It compares their
sum with the same number of nominal periods, using the same excess threshold.
Comparisons round the total threshold to the nearest nanosecond. For example,
`thresholdFrames=0.5` warns at a single interval of 1.5 periods or an integrated
span of `round(FPS) + 0.5` periods.

`frame_delay_enable` controls warning delivery for `atime`, `arrival`, `write`
and their `_sum` versions. `frame_delay_warning` reports the stored latches.
The first enabled crossing logs a warning and records `telem_ocamdelay`.
Acquisition restarting after a successful reconfiguration clears the latches.
Disabling a check suppresses its labrules and rtimv warning immediately while
cadence telemetry and history continue. Re-enabling restores any stored latch.
The telemetry reports measured spans in nanoseconds, `-1` for unavailable
spans, and the rate and sample count used to derive the nominal spans.

The explicit `reconfigure.request` switch saves the preceding history, ending
at the operator's request. Mode, FPS, and sync changes and automatic recovery
do not produce dumps. Relative `dumpDir` values are resolved below the normal
camera log directory. The writer retains the latest five binary files and
reports the filename, row count, and diagnostic losses through the app log.
A second manual request while a dump is queued or being written still requests
camera reconfiguration. Its skipped dump is logged.

Export a dump offline using the script in this directory's parent:

```bash
python3 apps/ocam2KCtrl/scripts/frame-delay-to-csv.py camwfs-frame-delay-….bin > frames.csv
```

The CSV exposes all three clocks in nanoseconds, the raw camera frame number,
stream counter, acquisition outcome, and mode and rate context. A possible
clock step marks unusual behavior in the external timestamps. The arrival
clock is monotonic, while the other two timestamps are in their original
clock domains. Analyze changes within each clock or compare interval changes.
The monitor can miss a backlog that was already stable when the capture began.

The little-endian binary format begins with `OCAMFD01`, followed by nine
unsigned 64-bit values (request ID, monotonic cutoff, realtime request time,
last covered sequence, handoff losses, capacity evictions, row count, epoch
count, and history duration in ns). The epoch table is `<QQQdQQQ>` (first
sequence, session, inode, FPS, SDK mode, software bin X, software bin Y). Rows
are `<qqqQQII>` (DMA, arrival or retention anchor, write, sequence, counter,
raw camera frame number, flags). Flags contain the outcome in bits 0–1,
frame/counter/arrival validity in bits 2–4, and possible clock step in bit 5.

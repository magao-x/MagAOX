# Loop timing monitor

`loopTimingMonitor` observes camera-to-DM software latency by polling two shared
image timestamps. It reports `SDK entry - camera atime`, preserving the SDK
timestamp captured before the native call even when the DM publishes it later.
It never commands a source device or uses image semaphore notifications.

Copy the settings from [loopTimingMonitor.conf](loopTimingMonitor.conf) into an
instance configuration. Camera, DM and loop INDI device names are independent
of the camera and SDK stream names. Defaults are `camwfs`, `dmtweeter`, `holoop`,
and streams `camwfs`, `dmtweeter_sdk_start`.

The BMC controller requires its existing SDK worker and pipeline mode. Add
`startTimeStream=dmtweeter_sdk_start` to its `[sdk]` configuration. The publisher
is disabled when this setting is empty. It writes only from the ordered post
worker, after existing command finalization and before pipeline credit retirement.
See [BMC SDK worker documentation](../bmcCtrl/doc/sdk-worker.md) for details.
Configuration and instrument restarts are operator actions.

## Collection requirements

All three received INDI conditions must hold:

- `<camera.device>.fsm.state` is exactly `OPERATING`.
- `<dm.device>.fsm.state` is exactly `OPERATING`.
- `<loop.device>.loop_state.toggle` is `On`.

The monitor uses INDI publish/subscribe with no application-level property
polling or state-age expiry. Each condition is initially unknown; unknown,
malformed or negative observations disable collection. A received affirmative
state remains valid until a later update changes it. INDI silence alone does
not change the gate. An off/on transition invalidates queued observations even
when it occurs between stream polling scans. Reopening collection establishes
fresh stream baselines and starts an empty statistics window. Camera/SDK
timestamp freshness checks and statistics-window expiry remain in effect.

`monitor.maxFPS` defaults to **3600 Hz**. Its reciprocal is a strict maximum
latency, approximately **277.778 us** at the default. Actual adjacent camera
timestamps also bound each match. A jitter-shortened camera interval is not
rejected merely because it is shorter than `1 / maxFPS`.

This assumes one successful, relevant DM submission per camera frame, with
true latency below a frame period. Auxiliary/syncopation commands must be absent.
There is no propagated camera frame ID: a true 600 us delay at a 500 us period
can look like 100 us. Configuring a maximum latency cannot detect this alias.

## Clocks, polling and coverage

The controller's SDK stream `atime` is `CLOCK_MONOTONIC`; its `writetime` is the
later diagnostic publication time. Its single byte is unused, its reader
semaphore count is zero, and its immutable `DM_SDK_TIME=monotonic-v1` keyword
identifies this contract. Its counter counts successful diagnostic publications,
not camera frames. Failed/no-send/administrative SDK operations are excluded.

The monitor maps camera timestamps into MONOTONIC using bracketed reads of the
explicit `camera.clock` (`realtime`, default, or `tai`). `camera.quantizationUs`
adds timestamp truncation uncertainty; the default 1 us matches the inspected
ordinary OCAM/EDT driver mode. That stamp is a driver completion-processing
timestamp, not a hardware-latched exposure or physical mirror-response endpoint.
All streams must originate on the same host and share the declared clocks.

The polling worker performs bounded metadata reads and sends immutable records
through a fixed-capacity queue. Ordinary application logic performs association,
aggregation, INDI publication and telemetry. `monitor.pollSleepUs` defaults to
10 us; zero explicitly selects busy polling. It is a requested sleep, not a
wakeup guarantee. `monitor.threadPrio` and `monitor.cpuset` allow optional thread
placement; defaults use ordinary scheduling and no affinity. Startup fails if
the requested scheduling policy/priority cannot be acquired. CPU isolation must
be configured by the operator.

Pairing uses endpoint time rather than poll arrival order. It waits for the
next camera timestamp and a later SDK timestamp before classifying an interval,
so a late post publication is not mistaken for the next camera's command. Zero
or multiple SDK candidates, uncertain interval boundaries, malformed timestamps,
counter gaps, queue overflow, clock jumps and source recreation break or reject
associations. Latest-value polling cannot recover overwritten publications.
Gap counts describe missed observations, not proven physical camera frame loss.

Both streams are baselined after a continuity failure; the app clears its
retained statistics conservatively. On source silence, old samples expire from
the trailing window even without further updates. Stream files are checked for
replacement every 100 ms while enabled. Counter/timestamp checks operate on each
scan; existing camera metadata remains a best-effort snapshot protocol.

## Output and configuration

`loop_timing` reports latency `mean`, sample `stddev`, `count`, camera/SDK cadence
means and deviations, accepted latency `span`, newest-pair `age`, and `valid`.
Times are **seconds**. Standard deviation uses `n - 1`; it is unavailable below
two samples. Unavailable values are `-1`, with zero count/valid where appropriate.
`loop_timing_state` reports availability, source endpoints and camera clock.
`loop_timing_quality` reports missed publications, invalid/ambiguous/unmatched
observations, queue/clock failures, counter regressions, capacity loss and epochs.

Persistent telemetry uses the existing **`telem_fgtimings`** schema and normal
telemeter configuration/forced-record mechanism:

| Field pair | Population |
| --- | --- |
| `atime`, `atime_jitter` (`acq`) | Consecutive camera `atime` intervals in the original camera clock |
| `wtime`, `wtime_jitter` (`wrt`) | Consecutive SDK-entry intervals in MONOTONIC |
| `wmatime`, `wmatime_jitter` (`wma`) | Accepted camera-to-SDK latency after clock mapping |

Normal and forced records require all three populations to be valid, fresh and
currently allowed by the operating gate. The existing six-field schema contains
no quality or state flags; inspect the INDI quality/state properties alongside
telemetry. Do not sum overlapping rolling windows as independent samples.

| Setting | Default | Meaning |
| --- | --- | --- |
| `monitor.maxFPS` | 3600 | Declared fastest frame rate and reciprocal latency bound |
| `monitor.pollSleepUs` | 10 | Delay after each scan; zero continuously polls |
| `monitor.windowSec` | 2 | Trailing event-time window |
| `monitor.capacity` | 32768 | Bound for unread records and each retained population |
| `monitor.clockBracketUs` | 1 | Largest accepted clock-read bracket |
| `monitor.clockMaxAgeMs` | 100 | Largest mapped camera timestamp age |
| `camera.quantizationUs` | 1 | Additional camera truncation uncertainty |
| `monitor.threadPrio` / `monitor.cpuset` | 0 / empty | Optional worker scheduling/placement |

The former `state.pollSec` and `state.timeoutSec` settings have been removed;
remove the old `[state]` section when updating an existing instance.

Increase capacity for higher configured rates, longer windows or slow application
logic; loss remains explicitly counted rather than silently downsampling.
No new work runs before the DM SDK call, but post publication and polling consume
CPU/cache bandwidth and may delay pipeline credit retirement. Offline tests
verify behavior; production timing impact requires operator qualification.

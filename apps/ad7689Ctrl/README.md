# ad7689Ctrl

Standalone AD7689 acquisition for MagAO-X, following `mcp3208Ctrl`'s
`MagAOXApp`, `dev::frameGrabber`, and `dev::telemeter` architecture. Timer mode
and ImageStreamIO semaphore synchronization use the same ADC owner and ordered
scan implementation. No reference application or downstream consumer is changed.

## Build and offline checks

Build on a **non-instrument Linux development machine**, with the normal MagAO-X
dependencies, Linux spidev headers, pkg-config, and **libgpiod >= 2.0** development
headers/library. The C API is used; a Python GPIO package is insufficient.

```sh
make -C apps/ad7689Ctrl
make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/AD7689_test
make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/ad7689Sync_test
make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/ad7689Ctrl_test
apps/ad7689Ctrl/tests/AD7689_test
apps/ad7689Ctrl/tests/ad7689Sync_test
apps/ad7689Ctrl/tests/ad7689Ctrl_test
# Run native publication separately in a fresh, private shared-memory namespace:
task_shm_dir=$(mktemp -d /tmp/ad7689-shm.XXXXXX)
MILK_SHM_DIR="$task_shm_dir" apps/ad7689Ctrl/tests/ad7689Ctrl_test "[ad7689Ctrl-native]"
# Separate fresh-process test with simulated 2 kHz WFS time, real frameGrabber/reader:
MILK_SHM_DIR="$task_shm_dir" apps/ad7689Ctrl/tests/ad7689Ctrl_test "[ad7689Ctrl-ddspc-native]"
rmdir "$task_shm_dir"
```

The real driver also has a portable fake-platform target, requiring only a C++20
compiler. It never opens GPIO or SPI hardware:

```sh
make -C apps/ad7689Ctrl/tests test
make -C apps/ad7689Ctrl/tests sanitize
```

The driver test substitutes the platform calls around the actual implementation,
with an independent ADC configuration/EOC/sample model. The app suite substitutes
the device interface and tests the real application controls, timing, and streaming
methods. Test-only GPIO/SPI headers apply only to the driver suite. Production
builds require the real Linux and libgpiod headers. Validation translation units
use `-fno-fast-math` to retain NaN/Inf checks.

## Confirmed initial profile

| Signal | Connection / setting |
| --- | --- |
| CNV | GPIO8, physical pin 24, persistent manual push-pull output, initially low |
| SCK | GPIO11, pin 23 |
| DIN | GPIO10/MOSI, pin 19 |
| SDO | GPIO9/MISO, pin 21 |
| SPI | `/dev/spidev0.0`, mode 0, MSB first, eight-bit words (current compatibility profile omits `SPI_NO_CS`) |
| GPIO endpoint | `/dev/gpiochip0`, chip-local offset 8; verify the target mapping |
| REF | External, pin 2, reported approximately 3.32 V |
| REFIN | Pin 3, reported approximately 3.32 V; verify the schematic/source |
| Supplies | Reported VDD/VIO approximately 3.3 V; common ground; COM grounded |

GPIO8 has been observed reserved by `spi0 CS0`. The user must resolve that kernel
ownership/pinmux before hardware commissioning. The committed SPI-driver compatibility
fix removed `SPI_NO_CS`; this scheduler change preserves that backend. Verify that
the configured SPI endpoint's native chip-select behavior cannot disturb the manual
CNV wire. A busy GPIO causes a startup failure; the application does not take a line
from another owner. The earlier `SPI_NO_CS` design is retained in the plan as history.

The example config uses a separate instance (`ad7689Ctrl`), separate output stream
(`accad7689_test`), IN0/IN1, 2000 frames/s, 1 MHz SPI, and a 10 us conversion hold.
No service, site configuration, active `acc000` producer, or production consumer
is installed or switched by this change. After independent hardware approval,
the user can select the example using the normal MagAO-X `-n` and `-c` options.
Agents must not run this application or hardware commands on an instrument.

## ADC protocol and timing

The implementation follows the [AD7682/AD7689 Rev. K datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad7682_7689.pdf),
especially Tables 6 and 13 and Figures 39/41, and cross-checks the two historical
configurations against the [ADI RAC driver](https://github.com/analogdevicesinc/no-OS/blob/main/drivers/adc/ad7689/ad7689.c).
The scratch Python scripts are working hardware references, with their quarter
bandwidth setting deliberately replaced by full bandwidth for production.

Supported CFG fields are CFG=1, INCC=111 (unipolar relative to GND), selected
INx=0..7, BW=1 (full), REF=110 (external direct REF; internal reference, buffer,
and temperature sensor disabled), SEQ=00 (manual order), and RB=1 normally.
`CFG = 0x3c71 | (channel << 7)` is left-aligned into `CFG << 2` and transmitted
as two bytes. For example, IN0 sends `f1 c4`, IN1 `f3 c4`, and IN7 `ff c4`.
Receive exactly two bytes and decode `(rx[0] << 8) | rx[1]` as unsigned 16-bit.
There is no MCP3208 command byte or 12-bit mask.

Each useful conversion waits for the previous acquisition/cycle/quiet guards,
raises CNV, holds it high through conversion, lowers it, waits the SPI setup guard,
and transfers with CNV continuously low. At the reported 3.3 V operating point,
tCONV is at most 3.2 us, full-bandwidth tACQ is at least 1.8 us, and the cycle is
at least 5 us. Defaults add a 10 us high hold, 1 us low-to-SPI setup, and 1 us
post-SPI quiet interval. Guards use monotonic deadlines after successful GPIO/SPI
completion, so software delays extend intervals. No clocks occur while CNV is high.
A fallback relative high hold protects cleanup after a timing-clock failure.
Actual edge timing and the part/voltage/temperature grade need physical validation.

Call k reads the result of its own CNV edge. The CFG sent in that call takes effect
at a later EOC and controls conversion k+2. Startup conservatively discards twelve
conversions, including delayed 32-clock CFG readback and restoration of 16-clock
framing. Unknown-state cycles are at least 20 us. A two-entry active/pending CFG
history primes the first two ordered channels; while reading slot i, the driver
writes the configuration for slot `(i+2)%N`. Normal frames use exactly N useful
conversions for N enabled channels, with no recurring dummy conversion. Reopen or
reconfiguration repeats initialization. SPI/GPIO/readback failures invalidate the
history and entire staged frame. Three consecutive failed acquisition/recovery
attempts stop acquisition; legal constant rail codes alone are not errors.

## Configuration and stream contract

`config/example.conf` lists all device settings plus inherited timing settings.
`accel.numChannels=1..8` selects the prefix IN0..IN(N-1). Optional
`accel.channels=[7,0,4]` selects an ordered, unique list. An explicitly supplied
count must match that list; runtime count changes cannot change its length.
`accel.fps` is the frame rate and the per-enabled-channel sample rate. SPI/CNV
and the explicit mapping are startup-only settings; runtime rate/count requests
are validated and applied at a stopped frame boundary.

The frame is N-by-1 `UINT16`, in configured channel order, with full codes
0..65535. `adc.vref` is calibration metadata for the conversion helper; it does
not rescale the raw stream. ADC-pin voltage is `D * VREF / 65536`, so 32768
corresponds to 1.66 V at VREF=3.32 V. Sensor acceleration requires separate zero,
gain, and sensitivity calibration.

Both modes set ImageStreamIO `atime` to software scan-start immediately before
the first useful conversion sequence. This precedes the first GPIO-high request;
scheduling/guard/syscall delays separate it from the physical aperture. Channels
are sequential samples, not simultaneous. Framegrabber maintains publication
`writetime`, `cnt0/cnt1`, circular records, timestamp arrays and semaphores. No
individual-channel timestamps are added. `timingDiag.channel_readout_us` and
`adcHealth.scan_us` report scan service duration, not physical per-channel offsets.
Measure aperture skew and software-timestamp offset externally.

Existing INDI `fps`, `numChannels`, `alpha`, `synchroDelay`, source subscriptions,
and `timingDiag` names/units are retained. `channelMap.inputs` reports ordered
physical inputs. `adcHealth` adds conversions, warmup discards, hardware errors,
invalid frames, overruns, missed timer/synchronized slots, recovery attempts, useful
published conversions, and scan time.
Diagnostic snapshots cross the acquisition/control thread boundary under a small
mutex; GPIO/SPI and semaphore operations run outside it. Invalid controls leave
current settings intact. Only complete fresh frames advance publication counters.

The unchanged `accelFFCtrl` currently assumes two channels and rejects raw values
above 4095. Its range/calibration extension is a separate companion task. Do not
connect this producer to it before that task and consumer qualification. Preserve
a two-channel profile for that handoff and record mapping, VREF, timing, and scale.

## Rate admission and hardware qualification

A requested rate must fit the configured lower bound
`N * max(high + setup + 16/SPIclock + quiet, 5 us)`; synchronized mode also reserves
`cadenceGuard_ns` (default 20 us). Actual Linux ioctl, scheduling, scan, logging,
and publication overhead are additional. The admission test cannot prove throughput.

| Profile | Configured eight-input scan budget | 8 kHz per channel |
| --- | --- | --- |
| 1 MHz, 10 us high, 1 us setup/quiet | 224 us | Rejected |
| 2 MHz, 10 us high, 1 us setup/quiet | 160 us | Rejected |
| 4 MHz, 4 us high, 1 us setup/quiet | 80 us | Candidate requiring measurements |
| 8 MHz, 4 us high, 1 us setup/quiet | 64 us | Candidate requiring measurements |

## WFS synchronization with multiple scans

Set `accel.fps` to the requested ADC rate and `synchro.shmimName` to the existing WFS
trigger stream. WFS FPS (`synchro.referenceFps` or the source subscription) and
requested ACC FPS remain independent. Both synchronization modes select a generic
positive integer ratio; no 2 kHz or four-scan assumption is built into the scheduler.
FrameGrabber publishes each complete scan as its own N-by-1 frame.

| WFS rate | Requested ADC rate | Scans per WFS period | Scan spacing | Scheduled ADC rate |
| --- | --- | --- | --- | --- |
| 2000 Hz | 8000 Hz | 4 | 125 us | 8000 Hz |
| 2000 Hz | 6000 Hz | 3 | 166.667 us | 6000 Hz |
| 2000 Hz | 4000 Hz | 2 | 250 us | 4000 Hz |
| 2000 Hz | 2000 Hz | 1 | 500 us | 2000 Hz |
| 1000 Hz | 8000 Hz | 8 | 125 us | 8000 Hz |
| 1000 Hz | 6000 Hz | 6 | 166.667 us | 6000 Hz |
| 1000 Hz | 3000 Hz | 3 | 333.333 us | 3000 Hz |
| 500 Hz | 8000 Hz | 16 | 125 us | 8000 Hz |
| 1800 Hz | 8000 Hz | 4 | 138.889 us | 7200 Hz |

The policy selects `max(1, round(requested_fps * measured_WFS_period))` scans per
WFS period. A noninteger ratio therefore produces the nearest integer-multiple
rate, reported in `syncTiming.scheduled_fps`; the ADC `fps` control retains the
requested target. The rounded candidate must satisfy the ADC service/guard budget
and the 10 kHz admission ceiling, otherwise that period produces no frames and
increments `adcHealth.overruns`. It does not silently reduce the scan count to fit.
Requested ADC rates below WFS cadence likewise round up to one scan per WFS period.

`synchro.referenceFps` is an optional initial WFS-rate hint. At its default zero,
two unique triggers establish cadence before legacy scans begin. DDSPC mode always
qualifies its rolling clock using `wfsFitMinEvents` observations, even with a hint.
In synchronized mode,
the existing `fps.device/property/element` subscription supplies this WFS hint and
**does not overwrite the ADC target**; in timer mode its previous ADC-rate behavior
is retained. In legacy mode measured triggers supersede the hint, modest changes
use the existing `synchro.alpha` EMA, and changes over 10% reset cadence immediately.
Legacy producer timestamps estimate cadence when consistent with monotonic arrival
intervals; counter gaps normalize missing WFS periods. DDSPC uses only the rolling
monotonic fit below; producer realtime timestamps and the cadence hint do not steer
its acquisition phase. Duplicate IDs do
not re-anchor or create slots, and queued notifications are coalesced and counted.

With `synchro.ddspcDelayEnabled=false` (default), each unique WFS reference arms
a finite schedule. The inherited WFS latency model and signed `synchroDelay` still specify a phase
within the WFS period. That phase is reduced modulo the ADC scan interval to find
the first scan's offset, so a member of the scan grid aligns with the modeled WFS
phase. For example, the default 2 kHz WFS model gives a 409.4 us phase: four scans
are nominally at 34.4, 159.4, 284.4, and 409.4 us after the trigger. The first offset
is capped when necessary to reserve ADC service and the cadence guard before the
next WFS boundary. The read-latency controller updates once per WFS period at the
scan nearest the model phase; its latency and all scheduling deadlines use the
monotonic clock. Stream `atime` remains the actual software scan-start realtime
stamp. Phase-error diagnostics wrap at the ADC interval for equivalent subscan phases.

In this legacy mode, waiting is interruptible by a new WFS notification. Each trigger replaces remaining
old-period slots; late or unsafe compressed slots are counted and skipped. Deadlines
are calculated from an absolute anchor and accumulated fractional intervals, so
6 kHz rounding does not drift. Source loss allows no scans beyond the last finite
WFS period. `synchro.timeout` (default 1 s) invalidates cadence after prolonged silence;
stream identity is checked at least every 100 ms or sooner for shorter timeouts.
Restart/reconfiguration re-primes the ADC and waits for a new reference. There is no
unbounded catch-up burst and no implicit free-running fallback when the source stops.

`syncTiming` reports `scans_per_wfs`, `scheduled_fps`, `scan_period_us`,
`wfs_period_us`, `phase_us`, `skipped_scans`, `late_scans`, `max_lateness_us`,
`cadence_resets`, `duplicate_triggers`, and `coalesced_wakes`. Source received/gap
counters remain WFS counters; published/written counters count ADC frames and
can exceed the number of WFS triggers. Software deadlines do not establish physical
CNV timing or sustained throughput; qualify jitter, phase, and losses on the board.

## DDSPC phase-aligned continuous acquisition

Opt in with the existing WFS stream and independent ACC target:

```ini
[accel]
fps=8000
[synchro]
shmimName=<RTC exposure-center reference stream>
referenceFps=2000
# DDSPC always waits for a qualified monotonic fit, regardless of this hint.
ddspcDelayEnabled=true
wfsFitMinEvents=16
wfsCadenceTolerance=0.1
wfsProcessTime_ns=75000
accLatency_ns=15000
ddspcMargin_ns=10000
ddspcSlewFraction=0.05
postDelay=0
```

The unwrapped first-acquisition target is
`T_WFS/2 + wfsProcessTime_ns - accLatency_ns - ddspcMargin_ns`.
The existing signed `postDelay`/`synchroDelay` offset is added **once**. The new
`wfsProcessTime_ns` is processing **after exposure completion**; the old
`wfsProcess_ns`, `dtTransfer_ns`, `dtF_ns` and `wfsRead_ns` terms belong exclusively
to the disabled/legacy model. They are not also added to this formula. The ACC
latency is an operator-calibrated scan-through-publication estimate for the actual
channel/SPI/load profile. The default 15 us is an illustrative model value, not a
measured capability of the example's conservative ADC settings. Physical admission
still checks the complete configured scan independently; measure publication
latency and choose a suitable margin before relying on DDSPC availability.

The following **nominal start deadlines** use processing 75 us, ACC latency 15 us,
and margin 10 us. Times are relative to the fitted reference of the first qualified
WFS frame; startup qualification precedes acquisition:

| WFS / ACC | Delay | Reference 0 slots | Reference 1 slots | Reference 1 arrives |
| --- | --- | --- | --- | --- |
| 2 / 8 kHz | 300 us | 300, 425, 550, 675 us | 800, 925, 1050, 1175 us | 500 us |
| 1 / 8 kHz | 550 us | 550, 675, 800, 925, 1050, 1175, 1300, 1425 us | 1550, 1675, 1800, 1925, 2050, 2175, 2300, 2425 us | 1000 us |

One acquisition owner tracks WFS references and serves a single continuous ACC
clock. Each accepted reference authorizes N further slot indices; bounded metadata
records retain each slot's original WFS ID, fitted phase reference and position, even after newer
semaphores arrive. These records do not create independent group schedules.
Normal references preserve the frozen pending deadline and extend authorization,
so 550/675 us slots survive the 500 us semaphore. No modulo wrapping or end-of-WFS
cap is applied to the DDSPC delay. A missing next reference allows the already
authorized delayed tail to finish, then suspends acquisition. There is no unbounded
free-running fallback, fabricated sample, or catch-up burst.

Deadlines use `CLOCK_MONOTONIC`, an unrounded segment origin and fractional-period
multiplication. New phase estimates and runtime delay changes move only subsequent
intervals, by at most `ddspcSlewFraction` of an ACC period (default 5%, maximum 25%),
also limited by scan/guard headroom. A reference never moves the pending deadline.
During an intentional slew the existing latency integrator is held; once settled,
its first-slot feedback provides one bounded correction to the DDSPC target. It
uses the slot's original reference, not the most recent WFS wake. Phase adjustments
can change transition intervals within those limits; steady nominal intervals
remain uniform. This is software scheduling, not a hard realtime guarantee.

DDSPC cadence and phase come from a ten-second rolling linear regression of local
monotonic arrival timestamps against **source frame IDs**. This retains approximately
20,000 observations at 2 kHz, 10,000 at 1 kHz, or 5,000 at 500 Hz. Timestamp/frame
origins move with the window; incremental centered residual moments support constant
time insertion/removal and avoid large absolute timestamps in the regression.
The fitted phase at the current source ID, never the raw individual arrival, feeds
both the continuous schedule and its existing delay feedback. Raw arrivals serve
liveness/outlier detection and diagnostics. No second PLL or competing integrator
is introduced: the rolling fit supplies low-bandwidth targets to the existing
bounded future-only correction. The interval remains `fitted_WFS_period/N`, with
fractional-period accumulation preserved. Its estimate changes gradually as the
clock drifts; it is not forced to follow +/-10 us source arrival errors.

The window fills progressively. `wfsFitMinEvents` (default 16, range 2..4096)
consistent unique observations qualify startup; this is about 7.5 ms from first to
sixteenth observation at 2 kHz, not a ten-second startup wait. Early estimates have
less noise rejection until the window matures. Choose a count that fits within ten
seconds at the source rate; known incompatible startup hints are rejected.
`wfsCadenceTolerance` (default 0.1, range 0.01..0.5) sets the relative phase/cadence
change gate, augmented by a 20 us noise floor and six times current inlier jitter
RMS; the candidate interarrival consistency gate has a 40 us floor to cover the
full difference of successive +/-10 us errors. Ordinary fluctuations remain in the fit. Larger individual outliers
are retained in history and excluded from regression; rejected observations do not
authorize new slots. While a longer qualification is pending, three consistent outlying observations
suspend the old ADC schedule. Qualification completes at the configured count. Once
confirmed, the new model contains only that new consistent sequence: old-rate
history cannot contaminate it. An isolated outlier followed by normal timing clears
the candidate without replacing the fit. Excluded intervals and deliberately
unstarted warmup periods are distinct from skipped already-authorized ADC deadlines.

Metadata consistency checks, duplicate rejection, coalesced wakes and independent
requested ACC FPS remain shared with legacy mode. Counter gaps use their actual
indices in the regression, while the existing ADC-gap policy explicitly reacquires
its sampling epoch. Counter rollback, stale identity, timeout and mode changes clear
the fitted model. Persistent phase/cadence changes and changes of the rounded integer
ratio explicitly discard/count pending work and acquire a new sampling epoch.
A 2-to-1 kHz WFS step at an 8 kHz ACC request therefore changes N from 4 to 8 and
target delay from 300 to 550 us; the new steady interval remains 125 us, with an
explicitly reported reacquisition discontinuity. Source disappearance/recreation
requests the existing stopped reopen/re-prime path. `synchro.timeout` invalidates
cadence/phase and waits for fresh references without reopening a healthy ADC.
WFS silence is a hard synchronization timeout even if a very long configured
delayed tail is pending.

Expired slots are counted and skipped. Ordinary lateness is allowed only within
`min(cadenceGuard_ns, 0.25*T_ACC)`; future starts must also respect the complete
scan/guard and minimum half-period separation. Configured and effective rates must
both fit the ADC timing budget and the existing 10 kHz ceiling. WFS periods retain
the existing maximum 1 s; causal DDSPC delays must be finite and within [0,1 s].
Invalid measured-rate/phase combinations suspend publication and increment
`adcHealth.overruns`, with `ddspcTiming.valid=0`. Known impossible startup profiles
are rejected. Association storage is bounded to 16,384 pending WFS references;
exhaustion likewise invalidates the clock rather than growing without bound.

`ddspcSettings` is a read/write numeric INDI property with `enabled` (0/1),
`wfs_process_ns`, `acc_latency_ns`, `margin_ns`, and `slew_fraction`. Partial updates
are validated and staged atomically. New phase settings are used at the next unique
reference without an application restart or ADC re-prime. Toggling the mode
explicitly cancels/counts pending slots and qualifies a new rolling clock.
Invalid updates leave accepted settings intact and set ALERT. Rate/channel changes
retain the existing stopped reconfiguration boundary.

`ddspcTiming` reports formula `target_delay_us` (before user offset/feedback),
`applied_delay_us` (scheduled first-phase offset, excluding the slot's index times
its ACC period), `sync_correction_us`, residual `phase_error_us`, `enabled`, `valid`,
and `phase_acquisitions`. The snapshot also distinguishes `scheduled_mono_us`,
`start_mono_us`, `completion_mono_us`, `published_mono_us`, `start_lateness_us`, and
`availability_us`. Completion means the full scan returned; publication is measured
after frameGrabber posts the output semaphores. `start_mono_us` is the software
invocation, **not** a measured physical CNV edge. `reference_mono_us`, `reference_id`,
`reference_slot`, `nominal_slot`, and `frame_seq` identify the associated source and
last committed output count. In DDSPC mode `reference_mono_us` is now the fitted
reference for that source ID, with raw arrival error reported separately. A zero publication timestamp denotes an unpublished
attempt. These INDI snapshots describe the latest attempt/publication; they are not
retained per-frame association records. Existing UINT16 stream geometry, `cnt0/cnt1`,
realtime driver scan-start `atime`, publication `writetime`, semaphore posts and
circular-buffer timestamps remain unchanged for consumers.

`wfsClock` adds `valid`, `samples`, `history_s`, `fps`, `period_us`,
`phase_mono_us`, `phase_frame_id`, `jitter_rms_us`, `arrival_error_us`,
`phase_drift_us`, `cadence_events`, `rejected_events`, `fit_resets`, and
`applied_phase_correction_us` (the continuous clock's applied phase offset relative
to its current epoch).
`phase_mono_us` and its frame ID specify the rolling fitted clock. Jitter is the
population RMS of inlier linear-fit residuals, including any unmodeled curvature;
large rejected events have a separate count. Long-term drift is relative to the
first full ten-second model and is zero while warming, so startup fit convergence
is not presented as measured clock drift. `ddspcTiming.phase_error_us` and
`sync_correction_us` show ADC phase residual and delay-controller correction;
`syncTiming.phase_us` reports the commanded phase, and
`ddspcTiming.start_lateness_us` distinguishes actual software acquisition error from
`wfsClock.arrival_error_us`. `adcHealth.missed_periods`, `syncTiming.skipped_scans`,
`syncTiming.cadence_resets`, and `ddspcTiming.phase_acquisitions` retain loss and
synchronization-recovery accounting. History has a 200,001-observation safety bound,
above the ten-second requirement at the supported 10 kHz ceiling; exhaustion clears
the fit and requires qualification rather than growing indefinitely.

## Hardware and on-sky validation

Hardware validation and deployment are user-run, separate work:

1. Record OS/kernel/libgpiod versions, GPIO mapping/owner/pinmux and SPI clock.
   Confirm schematic, external REF/REFIN wiring, voltage tolerances/grade, grounding,
   and input drive/settling. Resolve the known GPIO8 reservation before startup.
2. Capture CNV, SCK, DIN, and SDO during startup, scans, reconfiguration and stop.
   Check high/cycle/acquisition/quiet/setup intervals and native chip-select behavior,
   CFG readback, ordered channels and exactly sixteen clocks per useful result.
   Measure software `atime` to CNV offset and each channel's aperture offset.
3. Apply independently measured safe DC levels at zero, quarter, half and near
   full scale; compare mean codes against `round(Vin/VREF*65536)` with clipping.
   Use distinct levels across inputs and alternate large steps to expose ordering
   and settling errors. Establish uncertainty-based tolerances before acceptance.
4. Start at the example profile; compare 2 MHz/10 us against the scripts' single
   input runs, explicitly recording the BW difference. Qualify 4/8 MHz and 4 us
   holds before claiming performance. For each 1/2/4/8-input profile claiming
   8 kHz per channel, collect at least 60 seconds idle and under representative
   load: mean 7920..8080 Hz, zero unexplained losses, and all explained losses,
   invalid frames, missed deadlines and discards counted. Report RMS/max/complete
   interval distribution, scan skew, GPIO/SPI/scan/publication time and CPU use.
   No unapproved numerical jitter/skew threshold is implied.
5. With a 2 kHz WFS reference, verify four scans at an 8 kHz ADC target and three at
   6 kHz. Measure CNV spacing and phase relative to the WFS notification, including
   the interval across WFS boundaries. Step the WFS cadence and check the reported
   integer count and actual rate. Exercise delayed/coalesced/duplicate triggers,
   source loss and recovery; confirm skipped slots are counted without catch-up
   bursts. In legacy mode acquisition stops after the last predicted WFS period;
   in DDSPC mode the authorized delayed tail completes and then acquisition stops.
   Check the 300, 425, 550, 675, 800 us cross-boundary sequence and the 1 kHz table
   above. Match every accepted slot to its original reference; verify that duplicate
   and normal later semaphores do not reset the pending deadline.
6. Test restart, transport failure/recovery, stream recreation, retained records,
   shutdown and semaphore pacing; confirm an independent reader sees full-width
   values and correct timestamp/counter behavior on `accad7689_test`. Consumer
   handover and production ownership remain separate approval steps.

If software-paced CNV cannot achieve an agreed profile, report the measured limit
and review hardware-paced/kernel/DMA alternatives separately. Eight-channel 8 kHz
operation is a target, not a software-only or datasheet-only qualification claim.

For on-sky acceptance, first qualify the same profile with an offline/local RTC
reference and an independent reader. Have the instrument operator verify that the
selected RTC semaphore really marks exposure center and measure exposure-end to
DDSPC-entry time. Calibrate full scan-to-publication latency under representative
load, including channel skew, then select the margin from measured jitter/tail
latency. Compare sample publication to DDSPC entry (the target is margin before
entry), measure phase across many WFS boundaries, and exercise 2-to-1 kHz cadence
steps, runtime delay changes, source loss and reacquisition. Start with downstream
feed-forward disabled and a separate test stream; consumer qualification and any
production handover require their own operator approval. No on-sky or hardware
operations are performed by this implementation or its offline tests.

For rolling-clock acceptance, log at least several complete ten-second windows and
compare `wfsClock.arrival_error_us`/`jitter_rms_us` against the independently measured
CNV/sample interval distribution and `ddspcTiming.start_lateness_us`. Verify that
individual +/-10 us source errors do not appear as matching ADC phase jumps, while
slow fitted drift is tracked with bounded corrections. Check the sample count/span
at 500, 1000 and 2000 Hz, warmup, isolated delayed events, counter gaps/duplicates,
persistent 2-to-1 kHz changes and requalification. Software tests establish nominal
scheduling behavior; physical ADC jitter and worst-case regression/scan/publication
service under real Pi load require independent hardware qualification.

# dmSyncopation debugging and optimization record

This record tracks operator observations, diagnosis, changes and verification during debugging and optimization. The [main engineering plan](dm-syncopation.md) retains the settled design and review answers; the [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents current operation and interfaces.

Current status (2026-10-25 America/Phoenix): the operator normal / paused /
normal comparison confirms a working serial-housekeeping pause with the AO loop
closed and all settings/workload unchanged. Serial attempts remain at 981
through the captured quiet interval and resume afterward. Driver `atime`
intervals narrow to **499–501 us**, while the normal files retain roughly
480–525 us tails. This supports the prior IRQ 405 shared-lock/combined-handler
attribution of large completion-stamp excursions to serial/device handling.

A separate change late in the paused capture increases conversion spread while
SDK-return timing remains steady. Publication deviation is **0.761 / 1.428 /
0.747 us** before/paused/after, so this run does not show overall RMS improvement.
The paused conversion variation remains unexplained. Comparisons use disjoint
summary populations because paused updates overlap. Production housekeeping
scheduling remains unchanged; physical frame-arrival timing remains unmeasured.

Earlier isolated camera/acquWFS measurements and
their limitations are preserved below; matched camera-publication-to-wake
latency remains unmeasured. Median-grid scheduling, per-step skips and targeted
extra-step wakes remain unimplemented; the early-arrival reserve remains undecided.
Five-second native and one-second wrapper captures have now demonstrated
complete kernel retention at their respective observed event rates.

Append subsequent investigations as dated entries, keeping operator evidence, local verification and outstanding instrument checks explicit. All agent work runs on the workstation; the operator performs instrument actions and supplies results.

The camera-source review continues in the separate
[OCAM acquisition/publication timing plan](ocam2k-acquisition-publication-timing.md).
It contains source findings, implementation/verification records and the
latest operator capture analysis.

## 2026-10-20 first operator-run instrument trial

The operator supplied two status snapshots from the first instrument trial. The requested count was **two steps** at an observed main rate of approximately **1999 Hz**, but reported `deliveredHz` was approximately **1902 Hz** and `active` was **0**. The trial did **not** sustain the requested main-plus-auxiliary cadence and does not establish successful timing qualification.

Deployment identifiers differ from the earlier example: the INDI device is `dmtweeter-synco` (hyphen), while the shared-memory timing stream is `dmtweeter_synco_timing` (underscores). The actual primary filesystem stream is identified by `dm01disp03`/`aol1_dmC`; bare `dmC` is a planning shorthand, not the primary filesystem name in this deployment. Keep the INDI device and image identities explicit rather than deriving one from the other.

| Reported field | First snapshot | Later snapshot | Change/interpretation |
| --- | --- | --- | --- |
| Confirmed auxiliary `delivered` | 248 | 1501 | +1253; auxiliary combinations occurred intermittently. This cumulative count alone supplies no sustained rate. |
| `faults` | 628 | 3637 | +3009 synchronization/resource invalidations. |
| `missed` | 864 | 7720 | +6856 original timing failures. |
| `notStarted` | 863 | 7719 | +6856 step-0 copies suppressed before starting. |
| `copiedLate` | 1 | 1 | No additional completed-late copies between the supplied snapshots. |
| `snapshotMaxUs` | 5.44 | 6.542 | Observed whole-snapshot maximum increased; this is not a worst-case bound. |
| `copyMaxUs` | 20.018 | 20.018 | No growth between these snapshots; already above the initial 10 us budget. This observed maximum is separate from the current runtime planning budget. |

The operator also confirmed that actual monitor TID **464688** was running **FF (`SCHED_FIFO`)**, RT priority **30**, on CPU **61**. This is live operator evidence of the requested scheduling policy/placement, rather than configuration intent alone; it rules out the monitor simply running under a non-RT policy or on a different CPU in this observation. It does not establish a worst-case wake-latency bound or exclude other timing/scheduling causes.

The matching growth of `missed` and `notStarted`, with `copiedLate` unchanged, points to repeated pre-copy step-0 suppression as the dominant reported failure during this interval. It does not identify the root cause or prove that a particular keepout bound was violated. No elapsed interval or complete timing/configuration trace was supplied with these counters, so do not infer a recovery duty cycle, auxiliary event rate or safe controller spacing from their differences.

These initial status snapshots do not classify every fault cause; the subsequent code-level diagnosis and local verification are recorded below. An operator-run retest remains pending. Preserve hard safety/reader windows while investigating preparation and recovery behavior; do not label a successful local build/test or an occasional auxiliary combination as sustained instrument qualification. Agents performed no instrument commands, live INDI actions or hardware steps, and made no commit for this investigation.

## 2026-10-20 scheduler diagnosis and local fixes

Code-level diagnosis found a step-0 feedback failure mode consistent with the dominant `notStarted` growth. The earlier completion-residual controller could move the requested copy start later when actual copying was faster than its conservative budget. That consumed the small preparation reserve; a later wake could then fail the full-budget admission check before any copy started. Excluding those rejected attempts from learning and clearing the learned wait during ordinary miss recovery prevented that loop from learning the observed start latency. This code-level explanation is not proof that every instrument fault had the same cause.

- Step-0 feedback now targets copy-attempt start latency, not completion slack. Its advance is nonnegative and starts at the configured maximum on a real reset. Valid unclamped wake observations may update it even after safe pre-copy rejection. Completion residuals remain available for analysis, hard cutoffs/reader holds remain binding, and ordinary missed-step/budget reacquisition retains the learned correction.
- If the reader-safe planned start plus the current copy budget is already later than the hard cutoff, reject/count that step before waiting. Do not wait into a known-impossible window and convert the expected next main arrival into a secondary primary-change fault; this changes failure handling, not the hard safety window.
- Primary observation now ignores unchanged-counter callbacks and counts them as `duplicatePosts`. A counter jump/reset is adopted once before reacquisition; queued semaphore tokens do not create new physical main events or repeatedly reopen acquisition for the same counter value.
- Added latched `health.lastFault` alongside the changing phase/reason. New `timing` elements split faults into `primaryFaults`, `resourceFaults`, `outputFaults` and `triggerFaults`; expose `duplicatePosts`, `step0WakeUs`, `step0WakeMaxUs`, `step0AdvanceUs`, current `copyBudgetUs`/`snapshotBudgetUs`, `lastPrimaryDelta`, `lastIntervalUs`, `lastOutputDelta` and `lastFaultAgeSec` (-1 before a latched fault). These distinguish warm-up/missed-copy behavior from actual synchronization, resource, combination or unsafe-trigger faults.
- The guide now distinguishes the deployed hyphenated INDI device from underscore-named timing images and documents actual primary filesystem names, the revised wait-control semantics and the new readbacks. No timing-record layout, provider handshake or keepout relaxation is introduced.
- Red/green verification reproduced all three new regressions against the frozen pre-fix header: **3 expected failing cases**, with 38 passing/3 failing assertions. The failures cover a fast-copy-budget passive cycle, lack of priming by cycle 10 with 5 us wake delay, and an unchanged-counter callback destroying the accepted main timestamp. A forced rebuild of the current-header full scheduler suite passed **548 assertions in 13 test cases**. The normal `make -C apps/dmSyncopation` app build also linked successfully.
- No operator retest has yet established that the requested cadence is sustained or that the high instrument fault count is resolved; instrument qualification remains pending. No agent instrument actions or commits were performed.

## 2026-10-20 test commit checkpoint

The user requested committing the scheduler fixes for testing. Package the scheduler, regression tests, main-plan updates and this debugging record together, followed by a documentation-only commit for the user guide, per AGENTS.md. The preceding entries describe the investigation before this commit checkpoint.

Local validation remains **548 assertions in 13 passing test cases**, three expected failing regressions against the pre-fix header, and a successful normal app build. No functional changes were made after that verification. Operator retesting and instrument timing qualification remain pending.

## 2026-10-20 operator retest: primary interval faults

After commits `424c3cad` (scheduler fixes/tests/engineering records) and `cef5fb1b` (guide), the operator supplied disabled and enabled snapshots from a restarted instance, monitor TID **586278**. The primary is `dm01disp03`, geometry 50x50, requested count two, keepout 115 us, guard 20 us and step-0 lead 20 us.

| Enabled readback | Value | Interpretation |
| --- | --- | --- |
| `active`, `feasible` | 0, 2 | The requested schedule fits the current planning budgets but is not sustaining operation. |
| `delivered`, `deliveredHz` | 294, 1811.60585253788 Hz | Some auxiliary combinations occurred; confirmed main-plus-auxiliary cadence remains below the requested rate. |
| `faults`, `primaryFaults` | 630, 630 | All counted faults are in the primary class. Other fault-class counters are zero; this does not prove every primary fault had the same subtype. |
| `health.lastFault`, `lastFaultAgeSec` | `primary interval anomaly`, 0.00631239 | The latest latched fault concerns a primary interval. |
| `mainHz`, `plannedHz` | 2002.43095117473, 4004.86190234945 Hz | Both derive from the internal filtered callback period. |
| `cameraHz` | 2000 Hz | Last INDI camera FPS report; the operator separately reports the camwfs stream is stable at 1999.99 Hz. |
| `lastIntervalUs`, `lastPrimaryDelta` | 497.269, 1 | Latest distinct primary observation, not a latched copy of the faulting interval. |
| `missed`, `notStarted`, `copiedLate` | 1, 0, 0 | The earlier repeated step-0 start suppression is absent from this snapshot; overall operation still fails. |
| `copyBudgetUs`, `copyMaxUs` | 11.71, 10.71 | Runtime budget enlarged beyond the observed maximum. |
| `snapshotBudgetUs`, `snapshotMaxUs` | 10, 6.072 | Snapshot maximum remains within its initial budget. |
| `step0AdvanceUs`, `step0WakeUs`, `step0WakeMaxUs` | 20, 1.555, 3.555 | Step-0 advance is at its configured maximum; valid observed wake/check lateness is a few microseconds. |
| `duplicatePosts` | 0 | No unchanged-counter callbacks were counted in this snapshot. |

Source inspection confirms `mainHz = 1e9 / m_period`. Normal sequential observations update `m_period = (7 * m_period + observed_interval) / 8`, where timestamps are monotonic readings on entry to the primary DM callback. This is a short scheduling-period estimate, not a long-window camera or primary-counter FPS measurement. AO processing and callback observation latency can vary even when the camera's average rate is stable. The camera report assists the cadence sanity check; it does not set this period.

The displayed 2002.430951 Hz corresponds to **499.393 us**, whereas 1999.99 Hz corresponds to approximately **500.002500 us**. Their difference is approximately **0.609500 us** (0.122% in frequency). A stable average camera rate does not establish that every DM publication or callback lies within the configured per-event timing bounds.

A further code defect was identified: when a sequential interval is outside the predicted plus/minus `tracking.earlyMainUs` window but within the broad accepted rate range, the fault branch replaces `m_period` with that single anomalous interval before reacquisition. An isolated timing disturbance can therefore shift the prediction and provoke additional faults. This conflicts with the intended exclusion of isolated delayed callbacks from rate learning. The supplied snapshot cannot determine how many faults are caused by this feedback versus actual arrival/observation excursions.

Follow-up work should separate a stable rate measurement from phase/early-arrival validation, avoid adopting a lone anomalous interval as a new period while retaining recovery for genuine rate changes, and latch the faulting interval/predicted period or residual so it survives later healthy observations. No functional change, new test run, commit or agent instrument action was made for this diagnostic entry.

## 2026-10-20 measured acquisition reference and jitter diagnostics

The user requested the measured acquisition-rate report `camwfs.fg_timing.acq_fps` as the period reference, together with acquisition jitter information. The supplied frame-grabber timing packet was:

| Element | Reported value | Microsecond interpretation |
| --- | --- | --- |
| `acq_fps` | 2000.00091766453 Hz | Reciprocal mean acquisition interval, approximately 499.999771 us. |
| `acq_jitter` | 4.69486811812549e-07 s | 0.469486812 us acquisition-interval standard deviation. |
| `acq_min` | 0.000494003295898438 s | 494.003295898 us shortest observed acquisition interval. |
| `acq_max` | 0.000506162643432617 s | 506.162643433 us longest observed acquisition interval. |
| `delta_aw` | 5.36938433827207e-05 s | 53.693843383 us reported acquisition-to-write delay. |
| `delta_aw_jitter` | 1.28503846029492e-06 s | 1.285038460 us acquisition-to-write delay standard deviation. |
| `write_fps` | 2000.00401788876 Hz | Reciprocal mean frame-grabber write interval. |
| `write_jitter` | 1.10156319198172e-06 s | 1.101563192 us write-interval standard deviation. |
| `write_min` | 0.000467300415039062 s | 467.300415039 us shortest observed write interval. |
| `write_max` | 0.000524520874023438 s | 524.520874023 us longest observed write interval. |

Source inspection of `frameGrabber::updateINDI()` confirms the interval/jitter/delay values are in **seconds**; FPS is the reciprocal of mean interval. The inspected local camwfs configuration uses an approximately two-second timing window. The small acquisition jitter and stable mean support using a measured acquisition period, but do not bound all downstream AO/DM publication and callback excursions. The reported camera write extrema themselves cover a wider range than the acquisition extrema. They are not measurements of the DM controller's spacing and do not justify relaxing the configured hard bounds.

The reference design now prefers a fresh measured camera period (default configurable device/property/element: `camwfs`, `fg_timing`, `acq_fps`); the requested `fps.current` is no longer the default. The primary DM counter still establishes each cycle's phase. Camera reports are eligible only while accepted-reply age is below five seconds and their rate is within the configured main-rate range. The app requests subscribed values once per second, outside the cycle mutex: frameGrabber uses `updateIfChanged`, so an unchanged value does not inherently provide a heartbeat. Freshness is evaluated with the current monotonic time under the cycle mutex, separately from the unchanged callback-entry phase timestamp, so a newer camera report cannot cause a one-frame false fallback when the callback waited for the lock. A refreshed property is not proof that acquisition is ongoing; genuine DM-counter progression/silence checks remain binding. The camera-period path assumes nominal 1:1 DM/camera cadence; intentional division requires disabling that reference and using the qualified primary-window/nominal fallback.

If the camera reference is unavailable/stale, qualify a primary window of `tracking.rateFrames` sequential broadly valid intervals, with interval range no greater than twice `tracking.earlyMainUs`, and use total elapsed time divided by interval count. Otherwise use the configured nominal rate. Remove the short callback EMA and adoption of one anomalous interval as a new period. Reference-source changes or selected-period jumps greater than the configured early-main allowance reset phase/output/feedback acquisition without incrementing a synchronization-fault counter. These rules do not change the genuine main reference, requested step count or hard copy/reader/trigger bounds.

New diagnostics distinguish selected period from actual measured rates: `health.periodSource`, `timing.cameraAgeSec`, `primaryMeasuredHz`, `cameraJitterUs`, `cameraMinUs` and `cameraMaxUs`. `mainHz` is the reciprocal of the selected integer-nanosecond period; `cameraHz` preserves the actual measured floating-point report. The supplied acquisition FPS rounds to a 500000 ns selected period (`mainHz=2000 Hz`), so a small numerical difference from `cameraHz=2000.00091766453 Hz` is expected from quantization alone. Acquisition statistics are converted to microseconds and are diagnostic, not qualified bounds. A valid FPS-only packet clears missing optional statistics to -1; a present optional statistic that is nonfinite, negative, outside the supported range or has supplied minimum greater than supplied maximum rejects the entire packet without refreshing the last valid report. Primary interval anomalies additionally latch `faultIntervalUs`, `faultPeriodUs` and `faultResidualUs` so later healthy observations cannot overwrite the evidence; a subsequent fault other than an interval anomaly clears those interval-specific values to zero.

Local regression work also exposed a validation defect: `IndiElement::get<double>()` could silently convert textual `nan`/`inf` to zero, and zero is valid for optional statistics. The camera callback now checks complete raw numeric conversion before finite/range checks, rejecting malformed/trailing text atomically rather than refreshing the rate with fabricated zero jitter. Regression coverage now includes malformed, nonfinite and trailing numeric text, whitespace and atomic rejection.

A forced fresh scheduler test rebuild passed **1630 assertions in 18 test cases** with checked numeric parsing. Added coverage includes malformed/partial/nonfinite packets, whitespace, packet atomicity and missing-stat reset, camera period stability under callback jitter/outliers, qualified fallback and genuine rate change/counter gaps, strict five-second expiry, future/report-lock freshness cases, and fault latching. The earlier 548-assertion/13-case result remains the historical verification of the preceding fix. The normal sequential `make -C apps/dmSyncopation` app build linked successfully. Final clang-format dry-run and whitespace checks also passed.

Operator pipeline jitter and sustained-cadence qualification remain unresolved; there has been no agent instrument action, commit or successful operator retest of this reference change.

## 2026-10-20 measured-reference commit and push checkpoint

The user requested committing and pushing the measured acquisition-reference changes to `origin/jrmales/dm-syncopation`. Package the scheduler, regression tests, main plan and debugging record together, followed by the user-guide update in a documentation-only commit, per AGENTS.md. The preceding entry describes verification before this checkpoint.

Validation remains **1630 assertions in 18 passing test cases**, a successful normal app build, and clean formatting/whitespace checks. No functional changes were made after verification. Operator retesting and instrument timing qualification remain pending.

## 2026-10-20 operator report: no INDI response after restart

After pushed commits `21004ce9` and `7b570844`, the operator reported that `dmtweeter-synco` did not respond to INDI. Follow-up clarified that it **never responded after restart, before the scheduler was enabled**. The operator reports process PID **602544** alive, with the reported threads in state **Sl+** and wait-channel readbacks **`-`**. These observations do not classify a deadlock, FIFO/socket wait or other startup failure; startup messages and more informative operator-supplied wait-state evidence remain pending. No agent queried the instrument.

This disabled-startup observation makes starvation behind an active full-cycle scheduler callback an unlikely explanation for the reported startup failure. Static review did not establish a registry/cycle-mutex ABBA lock cycle. A separate concrete exposure exists in the measured-camera input callback: it previously waited for `m_cycleMutex`, which can be held for a complete timed cycle, while executing on the sole INDI input thread. That can delay unrelated input handling during active scheduling, but is **not proven to explain this startup incident**. INDI `sendXml`/FIFO output backpressure remains another open hypothesis, not a confirmed diagnosis.

The scoped code change separates camera-report handoff from timed-cycle ownership. The input callback parses/validates the packet and updates only a short camera mailbox lock. The cycle/status path attempts a nonblocking mailbox read into its coherent `CameraReport`; if the mailbox is busy, it retains the previous coherent report rather than waiting. The input callback no longer waits for the full timed-cycle lock, and the RT cycle never waits for the mailbox. Cached age remains based on the report's original validated receipt time, not its later adoption time. Existing validation, units, period-source priority, five-second expiry and hard timing bounds are unchanged.

The operator subsequently confirmed that INDI was responding **without deploying the local mailbox change**. The supplied startup terminal output contained only the monitor-placement notice for `spare61` and `dm01disp03 smThreadExec`. A privileged process snapshot was taken **after responsiveness returned**:

| PID/TID | Policy / RT priority | CPU | Reported wait channel |
| --- | --- | --- | --- |
| 602544 (main) | TS | 79 | `hrtimer_nanosleep` |
| 602547 | TS | 15 | `hrtimer_nanosleep` |
| 602548 | TS | 30 | `hrtimer_nanosleep` |
| 602549 | FF / 30 | 61 | `futex_wait_queue` |
| 602550 | TS | 85 | `do_select` |

No listed thread was reported in a pipe-write wait in this snapshot. The futex wait does not identify its mutex, and a post-recovery snapshot cannot classify the initial delay or exclude earlier output backpressure. Recovery therefore cannot be attributed to the un-deployed local change; the startup symptom's cause remains unclassified.

Local blocking/coherency regression verification passed **1660 assertions in 20 test cases**, followed by a successful normal app build and clean formatting/whitespace checks. The preceding 1630-assertion/18-case result remains historical verification of the measured-reference change. The mailbox change addresses a concrete blocking exposure; it does not establish the cause of the observed startup delay or qualify sustained instrument cadence. At that verification checkpoint, the change had not been committed, pushed or deployed on the instrument. No agent instrument action was performed.

## 2026-10-20 camera-mailbox commit and push checkpoint

The user requested committing and pushing the camera-mailbox change to `origin/jrmales/dm-syncopation`. Package the scheduler, regression tests, main plan and this debugging record together, followed by the user-guide update in a documentation-only commit, per AGENTS.md. The preceding entry records the precommit investigation and independent instrument recovery.

Validation remains **1660 assertions in 20 passing test cases**, a successful normal app build, and clean formatting/whitespace checks. No functional changes were made after verification. The original startup delay remains unclassified, and operator retesting of this mailbox change and sustained-cadence qualification remain pending.

## 2026-10-20 post-mailbox restart and disabled readback

After the mailbox commits, the operator again received two consecutive `No dmtweeter-synco.*.* from localhost:7624` responses, then a normal disabled readback. The intervening process checks used old PID **602544** and returned only their header. They establish that the old PID was absent, not whether the new instance was running or blocked. The eventual monitor TID was **611151**; it is a thread ID, not an independently established main-process PID.

The recovered readback selected `health.periodSource=camera acquisition`, with `cameraHz=1999.99900983901`, `cameraAgeSec=0.79091665`, `mainHz=2000`, `plannedHz=4000`, and zero faults/misses. Acquisition jitter was 0.615467165 us, with interval extrema 494.003295898–505.924224854 us. `primaryMeasuredHz=2000.11600672839` is a separate short-window primary measurement and does not replace the selected camera period. The camera report rounds to a 500000 ns period, explaining the displayed 2000 Hz. The disabled readback does not qualify enabled cadence.

Source inspection identified existing bridge behavior that can explain immediate query failures: app INDI startup signals the passthrough restart (`libMagAOX/app/indiDriver.hpp`); `INDI/xindidriver/xindidriver.cpp` drains three FIFOs until each is quiet for one second, then sleeps two seconds before launching forwarding threads. `INDI/INDI/indiserver.c` also enforces a ten-second minimum interval between starts of a driver, and can skip requests directed to a restarting driver. `getINDI` defaults to a two-second timeout. Retry fresh queries after bridge startup rather than assuming a longer timeout resends a request skipped during restart. Continuous outgoing traffic, including the new one-second polling, could extend a quiet-based drain; no timestamped bridge logs establish whether that happened here. These source findings fit the reported sequence but do not measure or prove its exact cause. No startup-code change is justified by this readback alone.

## 2026-10-20 enabled timing disturbance and disabled comparison

A subsequent enabled snapshot used monitor TID **613273**, so it must not be combined with TID 611151's counters as one continuous run. The selected period now follows measured acquisition timing correctly: `cameraHz=1999.99781744991`, `mainHz=1999.996000008` (500001 ns), `plannedHz=3999.992000016`, and `health.periodSource=camera acquisition`.

Enabled operation nevertheless remained unarmed/acquiring, with 1242 primary faults and no output/resource/trigger faults. The latest latched `primary interval anomaly` was **555.008 us** against **500.001 us**, residual **+55.007 us**, beyond the configured 20 us allowance. `lastPrimaryDelta=1` and `duplicatePosts=0`; the later healthy `lastIntervalUs=495.456` is a different observation. `missed=1`, `notStarted=0`, `copiedLate=0`, copy maximum 11.842 us/budget 12.842 us and snapshot maximum 5.951 us/budget 10 us do not reproduce the earlier repeated step-0 start suppression. Auxiliary `delivered=1140` and confirmed `deliveredHz=1929.41968844031` show intermittent extras, not the requested sustained cadence.

The operator disabled the same instance, waited for fresh camera statistics and reported **no other changes**:

| Camera acquisition statistic | Enabled readback | After disabling, direct `camwfs.fg_timing` |
| --- | --- | --- |
| Mean rate | 1999.99781744991 Hz | 1999.99900983901 Hz |
| Interval standard deviation | 8.101606773 us | 0.600831874 us |
| Shortest interval | 255.107879639 us | 484.943389893 us |
| Longest interval | 744.819641113 us | 514.984130859 us |

The following synco readback still used TID 613273 and reported Off, `health.reason=disabled`, `mainHz=2000`, `deliveredHz=0`, and an adopted camera jitter of 0.606584503 us. Its slightly different camera extrema/rate are from a later report window. Cumulative `delivered=9293` and `primaryFaults=faults=10075` include activity between the initial enabled snapshot and disabling; the supplied observations do not provide an elapsed rate or imply those faults occurred after disabling. The retained latest interval fault was 559.357 us versus 500.001 us (+59.356 us), already 74.66957831 seconds old in the disabled readback. Missed/not-started/copied-late counts remained 1/0/0, and other fault classes remained zero.

The larger acquisition timing spread coincided with enabled synco operation in these snapshots. The subsequently reported EDT buffer backlog confounds this comparison: its state and onset were not independently tracked across the snapshots, and the observations cannot establish whether synco caused either the backlog or the larger spread. Camera `acq_*` values use EDT-provided DMA timestamps; they do not independently establish physical exposure-clock jitter or current-frame delivery. The primary interval is measured at callback entry and includes publication timing plus changes in wake/dispatch latency. Enabled-only work includes repeated at-most-10-us sleep checkpoints, shorter combined-output polls, mapped copies/posts, and extra downstream DM combinations/commands. Timer/CPU interference, CPU/SMT placement, downstream processing and camera backlog are candidates to distinguish, not established causes. Retain the configured hard bounds rather than widening them to accept the observed excursions.

The next operator-controlled comparison is a fully stopped synco process, followed by another fresh `camwfs.fg_timing` window, to distinguish disabled monitor/polling activity from the enabled workload. Ordinary shutdown retains auxiliary contributions. That comparison remains pending. No functional code change, new test run, commit or agent instrument action was performed for these observations.

## 2026-10-20 stopped baseline and recurring INDI definitions

The operator stopped the app using `xctrl shutdown dmtweeter-synco`; the command reported several waits before the tmux session ended. The subsequent direct `camwfs.fg_timing` query reported:

| Statistic | Stopped-process value |
| --- | --- |
| Acquisition FPS | 1999.99996375131 Hz |
| Acquisition interval standard deviation | 0.585991481 us |
| Acquisition interval minimum / maximum | 488.042831421 / 509.023666382 us |
| Acquisition-to-write mean / standard deviation | 53.619915141 / 1.618420259 us |
| Write FPS | 2000.00330245155 Hz |
| Write interval standard deviation | 1.554393963 us |
| Write interval minimum / maximum | 466.585159302 / 536.203384399 us |

Acquisition jitter is close to the preceding disabled-process baseline (0.600831874 us), while the enabled snapshot was 8.101606773 us. The subsequently reported EDT backlog prevents treating these as verified healthy-camera baselines or attributing the larger spread to enabled scheduling/downstream work. The numerical observations remain useful history; one stopped window does not establish a timing bound.

The operator additionally observed that `getINDI camwfs.fg_timing` never finished while synco was running. Source tracing confirms a concrete cause: `dmSyncopation::appLogic()` forcibly requested its subscribed properties once per second. Every camera `getProperties` response supplies a property definition, and indiserver broadcasts that definition to interested clients. Property-wide `getINDI` has a wildcard element and waits for a quiet interval; its default two-second alarm is restarted on each matching definition (`INDI/INDI/getINDI.c`, `listenINDI()`/`findDPE()`). One-second definition traffic therefore prevents normal query completion. This is recurring forced polling, rather than a new request generated by every received camera update; server subscriptions are deduplicated.

Remove the app-specific periodic request and its timestamp member. Retain the base class's initial, unresolved and restart subscription handling and ordinary incoming camera updates. The five-second valid-reply expiry remains unchanged: if all published values remain unchanged long enough to produce no update, use the existing qualified-primary-window/nominal fallback and reacquire on reference changes. Do not use property-definition traffic as an acquisition heartbeat. The camera mailbox and hard timing checks remain unchanged.

This change addresses the demonstrated query-timeout regression. Removing repeated outgoing definitions may also reduce traffic during the bridge's quiet-based startup drain, but startup delay and the enabled jitter increase still require operator verification.

Local verification passed **1660 assertions in 20 test cases**, including existing incoming-report, mailbox, strict-expiry, fallback and unchanged-bound checks, followed by a successful normal app build and clean clang-format/whitespace checks. Independent source review confirmed normal initial/unresolved/restart subscriptions remain registered and the app-specific periodic request is absent. The app fixture does not exercise live driver transport; query completion after deployment remains an operator check. No new test scaffolding, agent instrument action or commit was introduced.

## 2026-10-20 periodic-polling fix commit and push checkpoint

The user requested pushing the periodic-camera-polling fix to `origin/jrmales/dm-syncopation`. Package the scheduler change, main plan and debugging record together, followed by a documentation-only user-guide commit, per AGENTS.md.

Validation remains **1660 assertions in 20 passing test cases**, a successful normal app build and clean formatting/whitespace checks; no functional changes followed that verification. Operator confirmation of normal query completion and startup behavior remains pending. The enabled timing disturbance remains unresolved. No agent instrument action was performed.

## 2026-10-20 operator discovery: camwfs EDT circular-buffer backlog

The operator reported that camwfs had become **very far behind in the EDT circular buffer** and that the AO loop was unstable even at low gain. In follow-up, the operator clarified that the diagnosis came from loop instability and the condition was cleared with **reconfigure on camwfs**. No direct queue-depth or frame-age measurement was reported. The operator notes that the synco work may have caused it and expects it affected the preceding measurements. Backlog depth, onset, and overlap with each earlier measurement window remain unknown. Record possible synco causation as an open hypothesis; neither an independent camera fault nor synco responsibility is established.

Local source inspection explains why rate alone cannot establish frame freshness:

- `edtCamera::pdvAcquire()` obtains a returned image and DMA timestamp through `pdv_wait_last_image_timed()`, queues another image with `pdv_start_image()`, and passes the timestamp to frameGrabber. This wrapper does not measure queue occupancy or frame age. The API name does not establish the SDK's backlog-draining or timestamp semantics.
- `ocam2KCtrl::acquireAndCheckValid()` checks successive returned camera frame numbers, accepting the first frame as its baseline. A sequence of older consecutive frames can satisfy that check. The published shared-memory counter counts accepted publications; it does not directly report camera-buffer age.
- `frameGrabber::appLogic()` computes acquisition and write FPS/jitter from differences between successive timestamps. A steady buffering delay can leave both rates near 2 kHz with small jitter while increasing the age of data reaching the AO loop. That extra delay can reduce loop stability; the supplied report does not quantify its magnitude or isolate the cause of the earlier primary faults.
- The existing `fg_timing.delta_aw` is mean write time minus the returned acquisition timestamp, and `delta_aw_jitter` is its standard deviation. Interpret this as an age/latency diagnostic only with a verified EDT timestamp endpoint and compatible timebase. The earlier roughly 54 us values cannot independently prove the ring was current without that contract and a known backlog timeline.

The prior raw readbacks are retained, with their causal interpretation revised above. Begin the next comparison after the reported camwfs reconfigure, with a fresh statistics window and a stable-loop baseline; then compare synco disabled/enabled while monitoring loop stability and any available verified buffer-health/frame-latency indicator. Capture camera acquisition/write timing, primary faults/residuals and delivered cadence together. Loop stability is the reported diagnostic symptom; it is not a direct queue-depth measurement. A near-2000 Hz mean or small interval jitter alone is insufficient to declare the camera current. At this checkpoint, no numerical post-recovery measurements had yet been supplied.

The repeated INDI definitions remain an independently established code defect addressed by the pushed polling fix. This evidence does not call for relaxing timing bounds or changing the scheduler on the strength of the confounded trial. This entry changes only the debugging record; whitespace validation is appropriate and no code build/test rerun is needed. No commit or agent instrument action was performed.

## 2026-10-20 post-reconfigure retest: camera spread improves, primary faults persist

The operator supplied a restart and disabled/enabled transcript after the reported camwfs reconfigure and polling-fix push. `xctrl restart dmtweeter-synco` also stopped and restarted `isRTC`, with its reported five-second start delay. The synco monitor TID is **619862**. All shown queries return to a shell prompt, including `getINDI camwfs.fg_timing` while enabled, consistent with removing the continuous-definition polling. There is no query/startup duration measurement, and this sequence also restarted the INDI server.

Direct camera readbacks before and after enabling were:

| Statistic | Before enabling | After enabling |
| --- | --- | --- |
| Acquisition FPS | 1999.99900983901 Hz | 1999.99877136108 Hz |
| Acquisition interval standard deviation | 0.580785514 us | 0.461334137 us |
| Acquisition interval minimum / maximum | 488.996505737 / 509.977340698 us | 489.950180054 / 509.977340698 us |
| Acquisition-to-write mean / standard deviation | 53.862984045 / 1.450978373 us | 53.682992625 / 1.262600916 us |
| Write FPS | 1999.99805592761 Hz | 1999.99900983901 Hz |
| Write interval standard deviation | 1.322678863 us | 1.191892152 us |

The enabled synco readback adopted a different camera statistics window: `cameraHz=1999.99900983901`, `cameraAgeSec=1.299660119`, jitter 0.625276265 us and extrema 481.128692627–513.076782227 us. Its disabled readback had age 0.013003428 seconds, jitter 0.458516962 us and zero faults/deliveries. Disabled callbacks return before primary phase checks, so their zero fault count is not evidence of clean primary callback intervals. These windows do not reproduce the earlier 8.101606773 us acquisition jitter or approximately 255–745 us extrema. They support separating that earlier spread from the remaining primary observation problem; FPS/jitter alone still do not measure ring depth or establish the cause of the earlier backlog. A final camera packet in the paste exactly repeats the initial direct packet and supplies no independent timing/order information.

| Enabled synco field | Value / interpretation |
| --- | --- |
| Period reference | `camera acquisition`, `mainHz=2000`, `plannedHz=4000`, requested/feasible count 2/2. |
| Primary fault latch | `primary interval anomaly`: interval 555.169 us, predicted period 500 us, residual **+55.169 us**, age 0.005584337 seconds. |
| Fault counts | `primaryFaults=faults=1277`; output/resource/trigger faults zero. Latest normal interval 502.59 us and counter delta 1 are separate from the fault latch; duplicate posts zero. |
| Current phase | `active=0`, `health.reason=acquiring main progression`; sustained requested operation is not demonstrated. |
| Delivery | 1178 confirmed auxiliary deliveries accumulated; `deliveredHz=1930.59261213876` counts confirmed primary-plus-auxiliary combinations over the app's status interval, including acquisition/fault gaps. It is not an independent measurement of all physical main updates. |
| Copy/snapshot | Copy maximum/budget 11.892/12.892 us; snapshot maximum/budget 6.171/10 us. |
| Step-0 outcomes | Missed/not-started/copied-late 1/0/0; wake/check lateness latest/maximum 1.431/4.152 us, advance 20 us. |

Local sequencing review found no fixed 50–55 us delay that explains the next primary callback entry. `processImage()` records monotonic entry time before its cycle mutex/resource checks. The monitor performs metadata/state checks after semaphore return, but adds no explicit 55 us sleep. The initial output-baseline drain lasts until approximately entry+50 us and returns; for the 500 us two-step schedule, the extra event and its output watchdog ordinarily finish well before the next main, and step-0 copying must finish by its earlier cutoff. An overrun that observes the primary counter changing inside a wait has a different fault reason. These facts do not rule out scheduler/preemption delays or an unusually long preceding callback; the numeric proximity to the 50 us watchdog alone does not establish causation.

The remaining distinction is **primary publication timing versus callback observation timing**. The existing fault latch measures the latter. A useful next diagnostic would capture coherent primary publication-timestamp intervals and callback entry/exit or rearm timing alongside the faulting counter, retaining unavailable/inconsistent metadata as such and keeping timestamp diagnostics separate from the monotonic scheduling contract. Compare intervals within their own clock domains rather than subtracting unverified absolute metadata and monotonic timestamps. Existing deterministic tests bypass the actual monitor/semaphore wake and do not qualify host dispatch latency. Preserve the configured hard bounds.

In follow-up, the operator explicitly confirmed that **the AO loop remained stable while synco was enabled in this run**, then clarified that the gain was low enough to tolerate an extra frame or two. At 2 kHz those delays would correspond to approximately 0.5–1 ms; this is an illustration of what stability could hide, not a measured delay or upper bound. The earlier overt instability was not reproduced, but loop stability cannot exclude a smaller latency hit or residual buffer lag. It neither measures buffer depth nor proves that every auxiliary update was delivered. Direct timing evidence remains necessary. This is a record/source-review update; no new code, build/test rerun, commit or agent instrument action was performed.

## 2026-10-20 primary-fault diagnostics implementation

The user authorized adding the proposed publication/callback measurements to dmSyncopation. Keep the original monotonic callback-entry timestamp and all scheduling decisions, hard bounds, reference selection and publication behavior. An app-local wrapper measures callback body completion before releasing the existing cycle mutex on every normal return, including disabled/resource/duplicate paths. Its duration includes cycle-mutex wait but excludes final unlock and monitor bookkeeping. The immediately preceding invocation can be an ignored duplicate; distinct-publication history advances only for distinct counters.

For each distinct primary, read `writetime` once under write-flag/counter checks, with no retry or additional stream/worker. Validate positive normalized representable nanoseconds, and compare only increasing timestamps from consecutive non-wrapping counters. Invalid/inconsistent samples and resource reattachment break the publication baseline; a coherent gap/reset sample seeds the next baseline without reporting an interval across the discontinuity. Compare timestamp intervals within their own clock domains. Clock jumps or an unverified producer timestamp contract can affect interpretation; never use these metadata diagnostics to move phase or relax the scheduling contract.

Latch the new `timing` elements alongside `faultIntervalUs`/`faultPeriodUs` at the same primary interval anomaly:

| Element | Same-fault evidence |
| --- | --- |
| `faultPrimaryCounter` | Associated primary counter. |
| `faultPublishUs`, `faultPublishValid` | Coherent consecutive publication interval, or -1/0 when unavailable. |
| `faultObserveExtraUs` | Signed callback interval minus publication interval; consult the validity flag because -1 also denotes unavailable. |
| `faultPrevCallbackUs` | Immediately preceding callback body duration. |
| `faultCallbackGapUs` | Previous body completion to faulting callback entry. |
| `faultEntryToCounterUs` | Faulting callback entry to metadata sample, even when publication timing is unavailable. |

Unavailable durations are -1 us. Ordinary acquisition/missed-step recovery retains the latch; a later real fault of another kind clears the interval-specific context. A roughly 500 us publication interval with a +55 us observation excess points toward changing observation/rearm delay; a roughly 555 us publication interval places the excursion in the producer's timestamp sequence, subject to its clock contract. Previous runtime and completion gap help identify an overlong preceding callback. These intervals cannot measure a constant EDT backlog or absolute WFS-to-DM latency. No configuration, provider timing-record layout or aggregate telemetry schema change is needed.

During this work the operator additionally reported **dm01disp07 at about 55 Hz and combined dm01disp at 2053 Hz**. Channel 07 counters increment on extra triggers; passive step-0 copies leave them unchanged. Approximately 2 kHz main plus 55 Hz extras is consistent with the combined rate over possibly different measurement windows and with intermittent recovery. Intended sustained two-step operation would produce approximately 2 kHz channel-07 counters and 4 kHz combined output. The app's confirmed `deliveredHz` can omit untracked main events during acquisition, so it is not an independent measurement of every physical main update.

Local verification passed **2349 assertions in 25 test cases**, adding five diagnostic cases to the preceding 20-case suite. Coverage includes independent publication/observation clocks and signed attribution (555 us observed with 500/555/600 us publication intervals), unavailable metadata, torn write/counter checks, malformed/overflow/equal/backwards timestamps, counter skips/reset/wrap, resource replacement/reattachment, fault retention/clearing, disabled/duplicate/scheduled callback completion and deterministic cycle-lock delay. New observation fixtures refuse to rewind virtual time. An initial fixture crash during reattachment was traced to its synthetic acknowledgement dereferencing the intentionally closed combined handle; guarding that test-only acknowledgement fixed the fixture before the successful full rerun.

The normal app build and final clang-format/whitespace checks passed. Independent review confirmed unchanged cycle control flow and all 44 INDI timing names/value mappings, metadata validation and fault-context lifetime. The guide and settled plan document the new fields and limits. Tests run with isolated workstation-local streams and do not establish instrument wake latency, metadata clock semantics or sustained cadence. After deploying an updated binary, the operator can collect `getINDI dmtweeter-synco.timing` with matching `health` output to obtain the correlated fault fields. No configuration change, commit or agent instrument action was performed.

## 2026-10-20 primary-fault diagnostics commit and push checkpoint

The user requested pushing the verified diagnostics to `origin/jrmales/dm-syncopation`. Package the scheduler, five new test cases, settled plan and debugging record together, followed by the user-guide update in a documentation-only commit, per AGENTS.md.

Validation remains **2349 assertions in 25 passing test cases**, a successful normal app build, independent source review and clean formatting/whitespace checks. No functional changes followed verification. Instrument results using the new fault-correlated fields remain pending; no configuration change or agent instrument action is included.

## 2026-10-20 first diagnostic retest: primary publication excursion

The operator supplied a new run with monitor TID **633227**, enabled with two requested steps and the same 20 us guard/lead and 115 us keepout. The operator explicitly reports the AO loop at **high gain** and confirms no frame-behind condition in this run. This is stronger operator evidence than the preceding low-gain stability check, which could hide an extra frame or two.

Before enabling, the direct camera packet reported 1999.99900983901 Hz acquisition FPS, 0.470901480 us acquisition jitter, interval extrema 497.102737427–502.109527588 us, acquisition-to-write mean 53.694916535 us/jitter 1.508909035 us, and write FPS 1999.99829440538 with jitter 1.527268223 us. The enabled synco packet adopted another window: camera FPS 1999.99710201713, age 1.087494416 seconds, acquisition jitter 0.424730220 us and extrema 492.095947266–507.831573486 us. Camera statistics still do not directly measure the downstream primary publication endpoint.

The new fields correlate the same latched interval anomaly at primary counter **123134542**:

| Field | Value / interpretation |
| --- | --- |
| `faultPeriodUs` | 500.001 us selected camera-derived period. |
| `faultIntervalUs`, `faultResidualUs` | 562.813 us callback-entry interval; +62.812 us relative to the prediction. |
| `faultPublishUs`, `faultPublishValid` | **562.743 us**, valid; publication timestamp interval itself is +62.742 us relative to the prediction. |
| `faultObserveExtraUs` | **+0.070 us**: only 70 ns difference between callback and publication intervals. |
| `faultPrevCallbackUs`, `faultCallbackGapUs` | 451.223 us previous body duration plus 111.590 us completion-to-entry gap, summing to the observed 562.813 us. |
| `faultEntryToCounterUs` | 5.681 us entry-to-metadata-sample elapsed time, including internal checks/possible lock wait. |
| Fault class/age | 1109 primary faults, zero output/resource/trigger faults; latest interval fault age 0.002651608 seconds. |

For this event, essentially all the long observed interval is present in the primary `writetime` interval. Changing semaphore/dispatch observation lag adds only 0.070 us; the preceding callback did not remain busy until this entry. Its body finished approximately 48.778 us before the next nominal callback time. These are relative timing observations: they do not measure absolute publication-to-entry lag or establish the configured 5 us observation bound. The 5.681 us entry-to-sample field is internal elapsed time and must not be interpreted as that absolute lag. Indirect synco interference with producer scheduling, upstream processing or downstream load remains possible; this packet does not establish the cause of the producer timestamp excursion or classify all 1109 primary faults as the same subtype.

The app remains unprimed/acquiring (`active=0`, feasible/requested 2/2) with 998 accumulated auxiliary deliveries and software-confirmed `deliveredHz=1928.51583067258`, versus planned 3999.992000016 Hz. Latest normal primary interval 489.835 us, counter delta 1 and duplicate posts zero are separate from the fault latch. Missed/not-started/copied-late remain 1/0/0; copy maximum/budget are 11.341/12.341 us, snapshot maximum/budget 3.376/10 us, and latest/maximum step-0 wake lateness 4.186 us. These values do not demonstrate sustained two-step delivery.

Local source tracing supports interpreting `writetime` as a completion/publication endpoint: the user-identified normal dmC producer `MVMextractModes` publishes calculated CPU/GPU results through `processinfo_update_output_stream()`, which calls `ImageStreamIO_UpdateIm()`. The inspected ImageStreamIO implementation timestamps `writetime`, increments the counter, clears the write flag and posts notifications. Confirming the deployed writer/version/clock contract remains relevant. Existing MILK process/stream trace data already carries writer PID, upstream trigger identity/counter and process-start/stream-update timestamps; that is a useful next source for locating which AO pipeline stage contributes variation, without another synco timing-control change.

The operator was asked for a **passive-only `steps=1` comparison**, keeping AO gain and camera settings unchanged, with two status snapshots about ten seconds apart. Disable before changing steps and enable afterward. This retains passive preparation and timing checks while removing later-step work and extra DM triggers. Persistent publication faults would show that the later-step path is not required for them; improvement would implicate that path as a candidate, without separating its copy/wait work from downstream command load. Counters are cumulative, so compare increments and latch ages between the snapshots. The comparison was pending at this checkpoint; the response is recorded next.

This is a debugging-record/source-analysis update only. No scheduling changes, bound relaxation, new test/build run, commit or agent instrument action was performed.

## 2026-10-20 passive-only one-step comparison

The operator supplied the requested Off, `settings.steps=1`, On sequence and two subsequent readbacks from the same monitor TID **633227**. Both report enabled/requested count one, selected 2 kHz main rate and planned 2 kHz rate. There are no wall-clock timestamps establishing the exact interval between the two readbacks; do not treat the requested ten-second wait as a measured duration.

| Field | First one-step readback | Later one-step readback |
| --- | --- | --- |
| Current state | `active=0`, `disabled settings changed` | `active=1`, `armed` |
| Primary/total faults | 13914 | 14061 (**+147**) |
| Other fault classes / duplicate posts | All zero | All zero |
| Auxiliary `delivered` | 12568 | 12568 (unchanged, as expected with no extra triggers) |
| Confirmed `deliveredHz` | 0 | 1989.49138651204 Hz, now primary confirmations only |
| Fault age | 20.9098916 s | 0.366973524 s |
| Fault counter | 123548842 | 123639933 |
| Fault callback / publication interval | 550.069 / 549.818 us | **475.799 / 476.199 us** |
| Fault changing observation delay | +0.251 us | **-0.400 us** |
| Fault previous body / completion gap | 448.507 / 101.562 us | **51.207 / 424.592 us** |
| Fault entry-to-sample delay | 5.911 us | 5.36 us |
| Camera acquisition jitter | 0.552893355 us | 0.365197675 us |
| Step-0 advance / latest wake lateness | 20 / 0.674 us | 1.389 / 1.416 us |

The first readback retains an approximately 21-second-old fault while the current reason is still `disabled settings changed`. That retained long interval is not evidence of a new fault under one-step operation. The 147-count increase and recent new latch in the later readback establish faults after the change without relying on that old latch. The earlier two-step count of 1109 and this first count of 13914 include intervening runtime; their difference cannot be assigned to the one-step test.

The new one-step fault is an **early** primary observation: callback residual -24.201 us relative to 500 us, while publication timestamps themselves are -23.801 us early. That exceeds the configured 20 us early-main allowance. The observation/publication interval difference is only -0.400 us. Its preceding callback body lasted 51.207 us and ended 424.592 us before this callback; their sum matches the 475.799 us callback interval. Thus this event again points to primary publication-timestamp variation rather than a preceding callback staying busy through the next entry. The two-step long interval and this one-step short interval are different events; they must not be paired as one late/early sequence.

One-step operation can arm between faults, and the unchanged auxiliary delivery count confirms no accumulated extras during this comparison. Its observed step-0 advance converged toward the measured wake/check lateness. These snapshots do not establish a whole-run fault rate or qualified bounds, and the later near-2-kHz confirmation rate does not prove every main update was tracked. Copy/snapshot maxima and the 1/0/0 missed/not-started/copied-late counters were unchanged.

The later-step path and extra DM commands are not required for these primary faults. They may still increase publication variation in two-step operation. One-step mode retains passive-copy waits/checks and associated CPU work, so it is not an app-disabled baseline. The next upstream investigation can use the existing producer/stream trace and actual writer placement; a disabled-versus-enabled publication-interval comparison would distinguish baseline AO timing from possible indirect synco interference. No timing-bound change is justified by these two snapshots alone.

The operator was asked to disable synco, select `dm01disp03`/`aol1_dmC` in `milk-streamCTRL`, and use F5 then Right Arrow for the expanded process trace, sharing writer PID and upstream stream names/PIDs. These keys are verified in the inspected local viewer source; actual instrument use remains operator-run. The response establishing the producer chain is recorded next.

This entry changes only the debugging record. Whitespace validation passed; no new code, build/test rerun, commit or agent instrument action was performed.

## 2026-10-20 upstream process trace and writer identification

The operator supplied four trace rows for the selected primary stream. The viewer's stream label resolves `trigger_inode`, while the PID is the process writing the downstream output, not the process writing the named input. This interpretation is confirmed by the inspected `processinfo_update_output_stream()` and `streamCTRL_print_SPTRACE_details()` implementations: the publisher records its PID and trigger input in row zero, and copies the input's trace into later rows.

| Row | Reported trigger inode | Trigger/input stream | Writer PID | Trace-derived output |
| --- | --- | --- | --- | --- |
| 00 | 1614 | `aol1_modevalDM` | **373333** | Selected `dm01disp03` / `aol1_dmC` |
| 01 | 1600 | `aol1_modevalWFS` | **373141** | `aol1_modevalDM` |
| 02 | 1589 | `aol1_imWFS2` | **372992** | `aol1_modevalWFS` |
| 03 | 2411 | `camwfs` | **372805** | `aol1_imWFS2` |

The resulting traced path is `camwfs -> aol1_imWFS2 -> aol1_modevalWFS -> aol1_modevalDM -> dm01disp03`. Camera acquisition's own writer is not identified by these four rows. The supplied PIDs identify processes, not necessarily their active RT thread TIDs. The trace contains no timing measurements locating the excursion to one of these stages.

The operator has been asked, with synco disabled, for `ps -L` policy/RT-priority/current-CPU readbacks for all four processes; `taskset -pc` allowed-CPU lists for their main TIDs and synco monitor TID 633227; and CPU 61's SMT sibling list. CPU 61 was previously observed for synco, but the current allowed-CPU list is needed to verify present placement. A last-observed processor alone does not establish affinity, and different logical CPU numbers do not rule out sharing a physical core. If a producer's active processing thread is not its main TID, follow its actual thread's affinity rather than assuming the process-leader mask applies to every thread.

Only the debugging record was changed. Whitespace validation passed; no new code, test/build rerun, commit or agent instrument action was performed. Operator scheduling/affinity results were pending at this checkpoint and follow below.

## 2026-10-20 operator scheduling and affinity readbacks

With synco disabled as requested, the operator supplied actual process/thread scheduling and main-thread affinity readbacks:

| Process / main TID | Name | Actual policy / RT priority | Last-observed CPU | Allowed CPU list |
| --- | --- | --- | --- | --- |
| 372805 | `acquWFS-1` | FF / 49 | 45 | 45 |
| 372992 | `wfs2cmodeval-1` | FF / 48 | 45 | 45 |
| 373141 | `mfilt-1` | FF / 48 | 46 | 46 |
| 373333 | `mvalC2dm-1` | FF / 48 | 45 | 45 |
| Synco monitor TID 633227 | `dmSyncopation` | Not included in this `ps` selection | Not included | **61** |

The `wfs2cmodeval-1` process also showed helper TIDs **373005, 373007, 373008** (`cuda0000340000e`, another `wfs2cmodeval-1`, `cuda-EvtHandlr`), all FF/48 and last observed on CPU 45. The `mvalC2dm-1` process showed helper TIDs **373501, 373531, 373532** (`cuda0000300000d`, another `mvalC2dm-1`, `cuda-EvtHandlr`), likewise FF/48 on CPU 45. The supplied `taskset` checks queried the process leaders; worker affinity masks have not yet been supplied independently.

CPU 61's `thread_siblings_list` contains only **61**. Therefore the four verified AO main-thread affinity lists do not overlap synco's monitor affinity or its physical core. The simple explanation of direct monitor/main-thread CPU or SMT sharing is unsupported by this placement. Multiple existing AO stages and their observed CUDA workers share CPU 45, while the filter is on CPU 46; this may be an intentional pipeline arrangement and does not itself establish the cause of publication variation or justify moving threads. Indirect memory/cache/NUMA, GPU/driver/interrupt or other system-load effects remain possible, without evidence selecting one here.

The operator has been asked to query all threads of the two CUDA processes with `taskset -apc 372992` and `taskset -apc 373333`, then inspect existing **`milk-procCTRL` F5 Timing** rows for all four stages while synco remains disabled. If timing is OFF, report that rather than changing measurement or timing-limit settings. This uses existing producer instrumentation instead of another scheduler change.

Local viewer/source inspection clarifies what those values mean: ITER summarizes successive process-execution-start intervals; EXEC summarizes elapsed execution-start to execution-end time, after the normal input-trigger wait. EXEC can include scheduling or GPU waits during the computation, but not the preceding input semaphore wait. The inspected timing ring contains 100 samples (about 50 ms at 2 kHz), so several refreshes are more informative than one clean-looking min/max window. `milk-procCTRL` F4 additionally exposes trigger/missed-frame/timeout state. Expanded `milk-streamCTRL` traces show process-start timestamps and age, but do not display execution duration even though stream-update timestamps exist in the underlying trace.

Only the debugging record was changed. Whitespace validation passed; no configuration changes, new code, test/build rerun, commit or agent instrument action was performed. Worker-affinity and per-stage timing results were pending at this checkpoint and follow below.

## 2026-10-20 CUDA helper affinities and disabled ITER baseline

The operator's all-task affinity queries confirm **CPU 45 for every listed thread** of both CUDA processes: 372992/373005/373007/373008 and 373333/373501/373531/373532. Together with the main-thread readbacks, this rules out direct CPU or SMT-core sharing between these AO threads and synco monitor TID 633227 on CPU 61. It does not select or exclude indirect system-load mechanisms.

With synco still disabled, the operator reports ITER medians of **500.0 us** for the four pipeline processes. Typical displayed ranges are **495–502 us**, with occasional **485–520 us** ranges at `mfilt-1` and later stages. The user explicitly confirmed both the corrected 495–502 range (initially typed 595–502) and that these are **ITER**, not EXEC, values. No execution-duration summaries were supplied.

These observations establish process-start timing variation while auxiliary scheduling is disabled. Around a 500 us period, the broader displayed range spans approximately -15 to +20 us, already near the configured 20 us phase allowance on its late side. They do not yet reproduce the approximately 476/563 us primary publication intervals captured during enabled operation. The viewer summarizes successive execution-start times over a short rolling window, not the primary stream's publication intervals. The stage summaries are not synchronized to each other or to a synco fault, so they cannot establish that the filter introduced the variation, identify computation versus dispatch/input timing, or measure its worst-case tail. The disabled app also retains its primary monitor, so this is not a fully stopped-process baseline.

The next useful diagnostic should correlate upstream stage timing with the same anomalous primary event. Existing propagated stream traces offer a source for current/previous stage timestamps and input-counter identity; pairing those coherently with a fault would be more informative than treating unrelated short-window extrema as one event. That is a proposed follow-up, not an implementation change in this entry. Preserve the hard timing bounds while identifying the source and supported uncertainty.

Only the debugging record is changed. Whitespace validation passed; no new operator action was requested after the clarification, and no new code, test/build rerun, commit or agent instrument action was performed.

## 2026-10-21 direct DM controller timing telemetry plan

The operator reports that analysis of recorded `dm01disp` timestamps has shown
substantially lower jitter than the various MILK timing displays. Treat the viewer
summaries and earlier sampled fault context as measurements of their own specific
endpoints/windows, not as qualification of controller command timing or total
loop latency. No further synco timing-control change is proposed on that basis.

The requested next step is a plan for timing telemetry in `dev::dm`, following
`dev::frameGrabber`/`ocam2KCtrl`: reuse `telem_fgtimings` for semaphore receipt,
SDK-return cadence and paired elapsed time, and add detailed DM processing
statistics. The separate [DM timing telemetry plan](dm-timing-telemetry.md)
defines exact hooks, backend differences, bounded RT collection, validity and
tests. It also identifies the frame/clock provenance needed before a controller
measurement can be combined with camera timing into a true full-loop result.

This is a planning checkpoint only. Existing operator evidence is preserved;
no implementation, configuration change, build/test rerun, commit or agent
instrument action was performed.


## 2026-10-21 DM controller telemetry implementation

The approved [DM timing telemetry plan](dm-timing-telemetry.md) is implemented for
controller-local measurements. The hot path captures monotonic integer markers
and fixed-size events only; ordinary logic computes elapsed-window statistics,
publishes read-only INDI and records compatible `telem_fgtimings` plus detailed
`telem_dmtimings`. Power-off logic also ages/publishes timing so old data cannot
remain frozen as valid. The [user guide](../../../libMagAOX/app/dev/doc/dmTiming.md)
contains exact endpoints, configuration and INDI units/names.

This does not identify MILK viewer summaries with controller command timing or
establish camera-to-DM latency. IrisAO success semantics remain unqualified;
BMC/ALPAO/PI use their actual SDK/transport results. Local synthetic tests/builds
and their precise SDK limitations are recorded in the separate plan, together
with a workstation capture benchmark averaging approximately 0.193 us/event,
with zero capture allocations or timing-event drops. That benchmark is not an
instrument overhead or tail-latency qualification. Operator
comparison against recorded streams, instrumentation overhead qualification and
camera frame/clock provenance remain follow-ups before full-loop claims or
further timing-bound decisions. No agent instrument command, deployment,
configuration change or commit was performed.


## 2026-10-21 local timing telemetry commit checkpoint

The user requested commits before operator-run hardware testing. Functional DM
controller telemetry and its engineering record are committed as `09028bb2`
(`Add DM controller timing telemetry`), followed by separate documentation and
braces/spacing cleanup commits. The existing validation checkpoint is **3826
assertions across 58 cases**, passing ThreadSanitizer and normal library, PI,
dmSyncopation and logdump builds. BMC/ALPAO/IrisAO SDK-specific coverage uses local
software stubs; vendor ABI and instrument timing qualification remain pending.

Formatting preserves non-brace C++ tokens and preprocessor directives. No new
implementation changes, build/test rerun, push, deployment or agent instrument
action was performed as part of commit preparation. The operator will perform
hardware testing.

## 2026-10-21 first DM telemetry run and publication comparison

The operator installed the controller telemetry and confirmed continuous 2 kHz
AO operation. One detailed two-second window reports 2,741 receipt intervals
over 1.99590386 seconds: mean 728.166 us, sample jitter 1,265.568 us, minimum
124.295 us and maximum 25.20309 ms. Receipt-to-SDK mean is 108.448 us with
0.503 us jitter, while SDK duration is 101.269 us with 0.185 us jitter. Counters
show no timing-queue drops or observed input-counter gaps; these counters do not
prove that every upstream command was observed. A compatible line from the
preceding second reports receipt/send mean approximately 1.384 ms; it belongs
to a different rolling window. Both record formats use seconds.

At the operator's request, input-stream `writetime` is now sampled alongside
`cnt0`, adding `publicationInterval` statistics independently of receipt and SDK
cadence. The [engineering record](dm-timing-telemetry.md) and
[user guide](../../../libMagAOX/app/dev/doc/dmTiming.md) document pairing, clocks,
telemetry/INDI names and metadata consistency limits. The stream clock may be TAI;
only differences within that clock are used. Local dmcomb's write protocol leaves
a brief timestamp/counter association ambiguity, so the new metric is observational
and cannot qualify exact frame identity or publication-to-wake latency.

A new operator comparison of publication, receipt and send statistics from the
same summary is pending. No cause for the irregular receipts is established,
and no timing-control setting or instrument state is changed by this work.
Scoped workstation tests pass **1,204 assertions / 22 cases**, as do library,
PI-335 and logdump builds; the engineering record contains the exact coverage and
existing-warning qualifications. No commit, push, deployment or agent instrument
action was performed.

## 2026-10-21 concurrent Python GPU workload discovered

The operator reports that `nvidia-smi` shows Python analysis processes running
and notes that the reported timing disruption may therefore be real. This is a
new workload confounder, not a confirmed cause. Which GPU the analyses use, their
load and their overlap with the pasted timing windows have not been established.
Do not dismiss genuine command-publication/receipt irregularity solely because
the camera reports 2 kHz or the AO loop appears functional. Conversely, internally
consistent telemetry arithmetic does not independently validate the measurement.

The next useful operator comparison is the same 2 kHz loop and settings with the
analyses paused, recording detailed/compatible records emitted together and the
new publication metric when installed. Compare elapsed coverage and sample/quality
counts alongside cadence and jitter. An improvement would support a workload
association; telemetry validation and exact attribution would remain separate.
No process was inspected or stopped by an agent. This entry changes only the
debugging record; no code, test/build rerun, commit, push or instrument action was
performed.

## 2026-10-21 timing recovers in the Python-workload pause comparison

Following the discussion of pausing the other user's Python process, the operator
reports a large improvement and supplies 19 consecutive `dmtweeter` compatible
records, timestamped 20:19:00.662358724 through 20:19:18.684848776 UTC. The named
source is
`/opt/MagAOX/telem/dmtweeter/2026_10_21/dmtweeter_20261021193132468510705.bintel`.
Ranges below are across the reported rolling-window means/sample standard
deviations, not individual-event extrema or a combined standard deviation:

| Measurement | Window mean range (us) | Window sample-jitter range (us) |
| --- | --- | --- |
| Receipt interval (`acq`) | 499.997755–500.002803 | 6.99847–7.67596 |
| Successful SDK-return interval (`wrt`) | 499.997792–500.002435 | 6.94744–7.63072 |
| Receipt-to-SDK latency (`wma`) | 108.525572–108.544261 | 0.426717–0.466923 |

Both receipt and send cadence now agree with the expected 2 kHz rate. Compared
with the earlier roughly 0.728–1.384 ms interval means and 1.266–2.465 ms jitter
from different windows, this is a major change in observed event spacing.
Receipt-to-SDK latency remains close to the earlier approximately 108.448 us,
supporting relatively unchanged controller handling after receipt.

This comparison strongly supports a workload-associated timing disruption and
makes genuine earlier long intervals more credible. It does not identify GPU
execution versus CPU/driver/memory effects, independently prove every earlier
timestamp, or measure camera-to-DM latency. The roughly 7–8 us residual interval
jitter is a sample standard deviation, not a worst-case bound. These compatible
records do not contain event extrema, quality counts or the new publication
metric; a detailed same-summary comparison remains useful. No further code,
test/build rerun, commit, push or agent instrument action was performed; this
entry and the linked engineering note record the operator's evidence only.

## 2026-10-21 camera IRQ pinning and detailed residual timing

The operator reports pinning the camera IRQ to the camera's CPU and provides 15
compatible records at 20:26:04.199811290–20:26:18.217867294 UTC, followed by five
detailed records at 20:28:03.355931828–20:28:07.360640641 UTC from the same
`dmtweeter` telemetry file. The particular IRQ number/CPU are not specified.

The first set has receipt-interval sample jitter of **5.87200–6.80430 us**, below
the previous post-Python-pause **6.99847–7.67596 us**. The arithmetic average of
the displayed rolling-window standard deviations changes from 7.25481 to
6.31775 us, about 12.9% lower. These overlapping two-second summaries are neither
independent samples nor a pooled event distribution. The later detailed set has
receipt jitter of **7.31060–7.81507 us** and send jitter of **7.43400–7.95624 us**,
so these observations do not establish a sustained IRQ-pinning improvement.
Mean cadence remains consistent with 2 kHz.

Across the five detailed windows, the reported statistics are:

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Receipt interval | 499.999345–500.003511 | 7.31060–7.81507 | 454.428–572.382 |
| Successful SDK-return interval | 499.999392–500.003511 | 7.43400–7.95624 | 453.296–572.331 |
| Receipt-to-SDK latency | 108.531479–108.559934 | 0.452247–0.457235 | 107.503–111.981 |
| SDK duration | 101.370502–101.376151 | 0.107996–0.114372 | 101.111–102.373 |
| Measured callback work | 124.978558–125.029648 | 0.518868–0.528373 | 123.293–128.573 |

Receipt intervals therefore include excursions of about **-45.6/+72.4 us**
relative to a nominal 500 us interval despite the stable mean and smaller standard
deviation. These are interval deviations, not directly the synco prediction phase
residual or a qualified worst-case bound. Controller work after receipt remains
comparatively consistent; the dominant measured variability is in spacing before
the receipt endpoint. Publication versus wake-up effects are still unresolved.

Each detailed summary covers approximately two seconds with 3,998–3,999 receipt
intervals and approximately 4,000 command events; since-summary event deltas are
2,002 or 2,003 over approximately 1.001 seconds. Slight differences between metric
counts are expected because both endpoints must lie inside the elapsed-time window.
Queue occupancy is zero except for one newly queued event in the third summary;
there are no reported failures, partial/no-send/rejected events, duplicates,
counter gaps, invalid local timestamps or dropped timing events in these deltas.
Missing counter-observation coverage still limits what `counterGaps=0` establishes.

The detailed records have the original twelve metrics and no `publicationInterval`;
they do not yet exercise the new input-timestamp measurement. An operator run of
that build remains the useful publication/receipt comparison. Only this debugging
record and the engineering note change for these observations; no new code,
test/build rerun, commit, push or agent instrument action was performed.

## 2026-10-21 publication timing installed: variation precedes controller receipt

The operator supplies camera compatible telemetry at
20:46:09.487242309–20:46:25.894217912 UTC, DM compatible telemetry at
20:46:41.974271188–20:46:58.996762840 UTC, and nine detailed DM records at
20:47:07.007294734–20:47:15.017751105 UTC. The DM records now include
`publicationInterval`, establishing that the updated measurement is installed.
Reported sources are:

- `/opt/MagAOX/telem/camwfs/2026_10_21/camwfs_20261021194808830042330.bintel`
- `/opt/MagAOX/telem/dmtweeter/2026_10_21/dmtweeter_20261021204403769510932.bintel`

The nine camera summaries retain approximately 500 us acquisition/write means.
Acquisition interval sample jitter is **0.288261–0.646438 us**, write interval
jitter **1.84804–2.31055 us**, and acquisition-to-write latency mean
**51.547897–52.143258 us** with **1.42152–1.90975 us** jitter. These camera windows
precede the DM windows, so they are not measurements of identical frames/time
populations and cannot establish a stage-by-stage jitter budget.

The nine detailed DM summaries show:

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Observed publication interval | 499.993606–500.002165 | 5.735345–6.694835 | 434.862–584.574 |
| Receipt interval | 499.993581–500.002210 | 5.757785–6.713803 | 434.921–584.645 |
| Successful SDK-return interval | 499.982180–500.001967 | 5.344454–6.270590 | 434.971–584.594 |
| Receipt-to-SDK latency | 109.358012–109.668889 | 0.699239–0.909290 | 108.034–114.346 |
| SDK duration | 101.760376–102.065930 | 0.597585–0.846659 | 101.141–106.080 |
| Measured callback work | 125.636804–126.025231 | 0.758731–1.010678 | 123.974–131.037 |

Publication and receipt interval counts match at 3,998 or 3,999 per window, with
matching local coverage; their sample standard deviations differ by only about
0.012–0.024 us. This strongly supports timing variation already being present at
combined-stream publication. It makes controller wake scheduling a less likely
dominant explanation in these windows, but equal marginal means/standard
deviations do not determine publication-to-receipt delay jitter. Do not subtract
the standard deviations or variances to infer that delay without paired samples
and a qualified clock relationship. The metadata association limitations recorded
above still apply, and window extrema do not prove the same individual events
produce each extreme.

Observed publication intervals include approximately **-65.1/+84.6 us** excursions
around 500 us. Controller receipt-to-SDK work is still comparatively consistent,
although its approximately 0.70–0.91 us jitter is higher than the earlier
approximately 0.45 us. The compatible DM sequence also shows receipt-to-SDK mean
drifting from approximately 108.95 to 109.53 us; the detailed SDK means then decline
from approximately 102.07 to 101.76 us. No cause for that modest variation is
established, and none of these spans measures camera-to-DM or physical response.

The new detailed since-summary counters report approximately 2,002–2,003 events
per approximately 1.001 seconds and zero failed, partial, no-send, rejected,
duplicate, counter-gap, invalid-clock or dropped timing events. One snapshot has
one newly queued event; that is occupancy rather than loss. Publication/receipt
window populations match here, improving the association evidence beyond merely
observing zero gaps, without proving full upstream frame provenance.

The next useful upstream comparison is contemporaneous primary-command
(`dm01disp03` / `aol1_dmC`, as previously configured) and combined-command
(`dm01disp`) publication intervals. This can distinguish variation reaching dmcomb
from variation introduced between primary and combined publication. The current
camera/DM aggregates do not identify a particular intermediate AO stage or prove
an IRQ-pinning effect. Only this record and the engineering note change; no code,
test/build rerun, commit, push or agent instrument action was performed.

## 2026-10-21 independent primary cadence added to dmSyncopation

At the operator's request, the app now uses the existing bounded `dev::dmTiming`
collector for continuous primary publication and actual semaphore-receipt
statistics. Capture precedes cycle locking and resource/duplicate/enable decisions;
collection therefore remains active while disabled or acquiring and with missing
auxiliary resources. Only primary monitor attachment changes the timing session.
Existing scheduler bounds, prediction, fault decisions and fault-latched diagnostic
sampling are unchanged.

The [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents the
new startup-only `diagnostics.windowSec=2` / `diagnostics.maxEvents=20000` defaults,
zero-disable modes, `primary_timing` and `primary_timing_quality` INDI properties,
and optional primary table in `[dmsynco]` telemetry. Durations are seconds. Each
ordinary app pass updates/publishes independently of the cycle mutex and records
approximately once per second. If scheduler fields cannot be sampled without
waiting, the record carries fresh primary data and `schedule_valid=0`; readers
must ignore the unavailable scheduler placeholders. Old records remain readable;
rebuild readers to display primary data and honor the new validity flag.

Source clock, metadata pairing, window/count and provenance qualifications from
the DM-controller measurement also apply here. The new data is independent of
SDK-oriented compatible validity and does not measure cross-clock latency.
Operator measurements from this new primary-observer version are pending.

Workstation checks pass **2,625 assertions / 34 cases**, including the real
private-stream monitor/hook/callback/collector route and legacy/new logger
roundtrips, plus normal library, dmSyncopation and logdump builds. The
[engineering plan](dm-syncopation.md) records exact coverage, build qualifications
and affected files. No new performance benchmark, commit, push, deployment or
agent instrument action was performed.

## 2026-10-21 primary cadence commit and push checkpoint

The user requested committing and pushing the verified primary observer to
`origin/jrmales/dm-syncopation`. Package implementation, tests and the main plan
first, followed by the guide/debugging notes and separate formatting cleanup.
The remote branch was current at preflight. Operator installation and a comparison
over overlapping primary/combined windows remain pending; no agent instrument
action is included.

## 2026-10-21 21:43 UTC combined-command validation sample

After the primary-observer push (`291c280e` implementation, `831d8d3d`
documentation, `e1c55ec9` formatting), the operator supplied eleven detailed
`dmtweeter` records at 21:43:08.434361143–21:43:18.446212508 UTC, session 2,
sequences 41–51. Reported source:
`/opt/MagAOX/telem/dmtweeter/2026_10_21/dmtweeter_20261021214223392496904.bintel`.
These are overlapping two-second windows updated approximately once per second,
not eleven independent trials. The attachment includes compatible DM `[fgtimes]`
records but no camera or synco `[dmsynco]` primary measurements. Its `enabled=1`
and `fg_valid=1` describe controller timing collection, not synco enablement or
installation of the new primary observer.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Combined publication interval | 499.998526–500.002244 | 7.301539–8.134835 | 422.788–577.070 |
| Controller receipt interval | 499.998514–500.002209 | 7.336856–8.176615 | 422.578–577.280 |
| Successful SDK-return interval | 499.998694–500.002352 | 7.325843–8.162254 | 423.460–576.749 |
| Receipt-to-SDK latency | 108.650605–108.667451 | 0.393606–0.417901 | 107.693–112.122 |
| SDK duration | 101.338286–101.342114 | 0.076175–0.095135 | 101.131–103.565 |
| Receipt through callback completion | 124.718938–124.759601 | 0.488494–0.526673 | 123.243–130.577 |

Publication/receipt interval populations have matching counts (3,998 or 3,999)
and local coverage in every record. Their jitter differs by only
0.028453–0.042349 us, while publication excursions reach approximately
-77.2/+77.1 us around the 500 us mean. The data again supports interval variation
already present at combined-stream publication. SDK-call variability is much
smaller and is not the dominant explanation in these windows. Compared with the
earlier supplied sample, publication jitter is higher, but workload/settings and
simultaneous upstream measurements are absent, so no cause is assigned.

All displayed failure, partial-send, no-send, rejected, duplicate, counter-gap,
invalid-clock and dropped-event totals/deltas are zero. A single initial
synthetic event explains the one-event excess of submissions over receipts.
Since-summary deltas contain 2,002–2,003 real receipts/submissions each; the total
receipt increase from the first to last record is 20,023 over approximately
10.012 seconds. This supports the expected 2 kHz cadence. Queue occupancy is zero
or one out of 20,000; occupancy is not loss. Sample counts can differ among
endpoint populations at elapsed-window boundaries without indicating loss.

The publication metadata association and clock qualifications from earlier
entries still apply. Matching marginal statistics do not measure
publication-to-receipt delay jitter or establish same-event correlations; do not
subtract their variances. The approximately 108.66 us receipt-to-SDK duration is
local controller latency, not camera-to-DM loop time or physical settling.
The approximately 0.176 us dispatch duration is monitor-to-callback bookkeeping,
not publication-to-wakeup latency.

The requested next comparison remains primary (`dm01disp03` / `aol1_dmC`) versus
combined (`dm01disp`) publication over overlapping windows. The operator was
asked for matching synco `[dmsynco]` records without changing current settings.
This update changes only this debugging record and the DM timing engineering
note. No code, build/test rerun, commit, push or agent instrument action is
included.

## 2026-10-21 synco reader sees records without the primary payload

The operator then supplied `[dmsynco]` records at 21:45:53.431636738,
21:46:04.434309866, 21:46:14.436789612 and 21:46:24.439273115 UTC, each ending
`schedule_valid=1 primary_timing=unavailable`. These identify the correct telemetry
type, but do not yet provide the new primary measurement. The updated reader emits
that exact message only when the optional primary table is physically absent.
The current producer always serializes the table, even when diagnostics are
disabled or samples are not yet valid. Disabled collection would instead show
`primary_enabled=0` and empty populations. The default-true `schedule_valid` also
applies to legacy records and does not establish that the emitter is updated.

The likely explanation is an older running/installed dmSyncopation binary, or
reading an older telemetry file. The roughly ten-second record cadence also fits
the earlier recording path, but does not by itself identify the process version.
The next operator check is the presence of
`dmtweeter-synco.primary_timing_quality` on INDI and confirmation that the followed
log is current. Rebuild/install/restart the synco app if the running version is
old; scheduling can remain disabled while collecting the new diagnostics.
No collection-setting or timing-bound change is indicated by these records.

## 2026-10-21 21:54 UTC primary and combined publication comparison

The operator confirmed that dmSyncopation had not been restarted. After restart,
eleven supplied synco records at 21:54:02.450023755–21:54:12.456341470 UTC now
contain `primary_enabled=1 primary_valid=1`, session 2, sequences 36–46. This
resolves the missing-payload observation. Ten controller records cover
21:54:01.579341360–21:54:10.591327192 UTC, session 3, sequences 689–698. The two
collectors' session numbers are independent and do not identify matching frames.
No source telemetry filename was included in these excerpts.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Primary publication interval | 499.999199–500.001076 | 3.411070–4.165429 | 435.984–580.386 |
| Synco primary receipt interval | 499.999317–500.001104 | 3.420770–4.167280 | 435.843–580.426 |
| Combined publication interval | 499.998458–500.002772 | 5.588990–6.808115 | 431.094–584.484 |
| Controller receipt interval | 499.998468–500.002812 | 5.596426–6.839504 | 430.914–584.715 |
| Successful SDK-return interval | 499.998788–500.002827 | 5.640614–6.879813 | 432.036–585.957 |
| Controller receipt-to-SDK latency | 108.637231–108.662032 | 0.417072–0.429258 | 107.683–112.874 |
| SDK duration | 101.336551–101.339361 | 0.084473–0.103309 | 101.121–104.577 |

Nine same-second synco/controller report pairs have reporting offsets of
0.130570–0.136236 seconds. Their nominal two-second windows therefore overlap
heavily (approximately 93% by report time), although they do not contain identical
commands or have identical endpoints. Combined publication jitter exceeds primary
publication jitter in all nine pairs. For example, the 21:54:03 pair reports
4.165429 us primary and 6.759755 us combined; the 21:54:10 pair reports 3.702233 us
and 5.588990 us respectively. Consecutive windows within each series also overlap;
these are not independent trials or a paired per-command latency measurement.

The contemporaneous population comparison supports variation already reaching
the primary and additional variation in the primary-to-combined publication
path. It narrows the next investigation to combiner notification/polling,
scheduling, summation, other-channel work and output publication. It does not
identify which component dominates. Controller receipt timing closely tracks
combined publication, and its local work remains much steadier. No independent
added-jitter value is calculated by subtracting standard deviations or variances;
the necessary per-command covariance/provenance is absent. External timestamp
clock and metadata-association qualifications from earlier entries still apply.

Each synco window has 3,999 publication and receipt intervals with matching
coverage. The collector has no queued events and zero rejected, duplicate,
counter-gap, invalid or dropped totals throughout; receipts increase by 20,012
over approximately 10.006 seconds. Controller publication/receipt counts likewise
match at 3,998–3,999, with zero reported failure/loss totals/deltas and queue
occupancy zero or one. Both mean cadences are consistent with 2 kHz.

Synco reports `state=0 steps=2 delivered=0 missed=0 faults=0` throughout.
`primary_enabled` describes diagnostics; `steps` is the configured request and
`schedule_valid=1` only means that scheduler fields were sampled. No active
schedule or extra delivery is demonstrated, and these fields do not independently
report the enable toggle. The primary population still has approximately
-64/+80 us excursions around 500 us; its roughly 4 us RMS does not qualify hard
timing bounds or show that every interval is safe.

The next useful measurement is a short sequence associating each primary command
with its combined publication, with a verified clock relationship and trigger
provenance, to measure the distribution and changes of primary-to-combined delay.
Combiner wake/start and publish endpoints could then locate its variable portion.
The current aggregates do not establish camera-to-DM latency or justify a timing
bound, polling or scheduling configuration change. This is a documentation-only
evidence update; no code, test/build rerun, commit, push or agent instrument action
is included.

## 2026-10-21 combiner trace observer implemented

The operator authorized building dmcomb-behavior measurements and considering
configuration changes. The app now independently monitors combined-output
metadata/process trace while retaining its existing primary collector and
scheduler. The observer uses a separate automatic semaphore/handle and ordinary
priority by default. It captures fixed observations without pixel copying,
allocation, logging, cycle locking or any producer/configuration mutation.
Ordinary app logic computes the same two-second default window and publishes
`combiner_timing`, `combiner_timing_quality` and an append-only optional combiner
table in `[dmsynco]` telemetry. Each of six populations has independent
count/coverage/variance availability; all durations are seconds.

The endpoints are consecutive combined publication/receipt intervals, iteration
execution start to trace update, trace update to ImageStreamIO writetime, and
linked primary trace update to execution start/output writetime. These use
existing producer trace endpoints: execution includes checks/optional work/sum,
publication excludes later notification posts, and upstream trace update precedes
primary writetime. They are not an isolated summation benchmark or proven atomic
per-frame latency. Positive normalized/ordered timestamps, stable repeated
metadata/trace reads and advancing writer/start/primary timestamp/counter
baselines qualify each duration. Missing, unlinked or stale trace data remains
unavailable with explicit quality counters; output cadence can remain usable.

Local source tracing confirms that naming the real primary in
`.procinfo.triggersname` while keeping DELAY mode 4 supplies upstream trace
linkage without switching away from all-channel polling. That startup setting
requires a dmcomb restart and adds trace-copy work. Actual mode/delay, processinfo
timing enablement, CPU placement and trace clock contract remain operator checks.
The inspected trace does not reliably report trigger mode, and its sampled input
counter does not prove which pixels were consumed. The
[guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) describes these limits,
the FPS controls and a controlled comparison. No live configuration is changed.

Scoped workstation checks pass **3,112 assertions / 49 cases** (collector 100/7,
app 2,690/35, logger 322/7), plus normal library/schema, app and logdump builds.
Tests include a finite real combined monitor and concurrent queue wraps. Review
corrected gap freshness and repeated upstream timestamps, both with regressions.
The [main plan](dm-syncopation.md) records file scope and detailed verification
qualifications. No new instrument-performance, total-loop-latency or physical
settling result is claimed. No commit, push or agent instrument action is
included; operator installation/measurement is next.

## 2026-10-23 combiner measurements commit and push checkpoint

The user authorized committing and pushing the verified combiner observer to
`origin/jrmales/dm-syncopation`. Fetch confirmed the branch was current, and the
three focused suites passed again: **3,112 assertions / 49 cases**. Formatting
and whitespace checks pass. Package the implementation and main plan first,
then the user guide/debugging notes and separate formatting cleanup. Operator
installation, clock/trace configuration checks and measurements remain pending;
no live configuration change or agent instrument action is included.

## 2026-10-23 19:24 UTC comparison still lacks combiner-stage output

The operator supplied thirteen synco records at
19:24:21.864556852–19:24:33.879001139 UTC (primary session 2, sequences
17,704–17,716) and twelve controller records at
19:24:23.335999571–19:24:34.351111872 UTC (session 5, sequences
17,683–17,694). No telemetry source filenames were included.

The synco lines contain valid primary populations but no combiner fields at all.
The current reader always appends either the combiner table's formatted fields or
`combiner_timing=unavailable`, even for legacy writer records. Complete absence
of both forms indicates an older formatter, or externally removed/truncated text;
it does not establish that the running app is old. A new writer decoded by an old
reader would have this appearance. The operator was asked for `type -a teldump`
and the read-only `dmtweeter-synco.combiner_timing_quality` property to distinguish
the reader command from the app's installed/running interface. No reconfiguration
is prescribed before resolving this missing-data issue.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Primary publication interval | 499.997938–500.002000 | 5.664070–6.225451 | 421.947–574.215 |
| Synco primary receipt interval | 499.997884–500.002088 | 5.718387–6.284060 | 421.897–574.385 |
| Combined publication interval | 499.996475–500.003297 | 7.253794–7.678544 | 421.867–576.348 |
| Controller receipt interval | 499.996535–500.003295 | 7.277665–7.699629 | 421.866–576.358 |
| Successful SDK-return interval | 499.996232–500.003433 | 7.290447–7.712534 | 421.787–575.517 |
| Controller receipt-to-SDK latency | 108.988375–109.138374 | 0.435409–0.524606 | 107.833–113.314 |
| SDK duration | 101.298183–101.303501 | 0.071630–0.083170 | 101.101–102.314 |

The mean cadence remains consistent with 2 kHz. Publication/receipt interval
counts and local coverage match within each observer: 3,999 primary intervals
and 3,998–3,999 combined intervals per nominal two-second window. Nearby report
pairs are offset by approximately 0.470–0.472 seconds, so their nominal windows
overlap by approximately 76%; commands remain unpaired. The larger combined
publication jitter is consistent with earlier evidence of additional variation
in the intervening path, but it does not identify a combiner stage, quantify
independent added jitter or establish total loop latency. Controller SDK duration
remains much steadier. Primary/combined interval tails still reach roughly
-78/+74–76 us about 500 us; RMS alone does not qualify timing bounds.

Primary cumulative duplicates/gaps are 1,061/1,060 throughout, with zero new
problem counts over 24,029 additional receipts. Controller cumulative
duplicates/gaps are 1,000,553/977,626 throughout; every displayed since-summary
delta has zero duplicates, gaps, failures, partial/no-send, rejected, invalid or
dropped events, with 2,002–2,003 receipts/submissions. Those large totals therefore
reflect earlier observations, not new anomalies in the displayed span. Their
onset/cause cannot be dated here; gap counts are anomalous counter transitions,
not an inferred number of missing frames. Queue occupancy is zero or one.
Synco remains unarmed in the snapshots, with zero deliveries/misses and a single
unchanged cumulative fault. The excerpt does not identify that fault's cause.

The combiner execution/publication/linkage measurements are still pending an
updated-reader readback. This evidence updates only the debugging record and DM
timing note; no code, test/build rerun, commit, push, live configuration or agent
instrument action is included.

## 2026-10-23 19:31 UTC combiner stages available; primary linkage absent

After identifying `logdump` as the reader requiring an update, the operator
supplied fifteen complete current-format records from
`dmtweeter-synco_20261023191911531081671.bintel`, at
19:31:02.354944804–19:31:16.369481373 UTC. Primary and combiner sessions are both
2, with sequences 18,104–18,118. The new combiner table is present and enabled;
execution/publication populations have 4,000 observations in every nominal
two-second window, and all four publication/receipt interval populations have
3,999. Thus the earlier missing-field issue is resolved in this readback.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Primary publication interval | 499.999398–500.001620 | 2.973495–4.544225 | 423.209–579.835 |
| Synco primary receipt interval | 499.999423–500.001675 | 3.006357–4.562379 | 423.109–579.534 |
| Combined publication interval | 499.999338–500.002665 | 6.118142–7.122052 | 421.577–578.111 |
| Combined observer receipt interval | 499.999330–500.002850 | 6.140706–7.818030 | 421.506–578.222 |
| Combiner execution start to trace update | 4.928640–5.048666 | 0.158043–0.265622 | 4.459–10.189 |
| Combiner trace update to writetime | 0.156957–0.165228 | 0.020552–0.167473 | 0.120–4.339 |

The approximately 5 us execution duration includes counter checks, optional work,
summation and trace setup. The short final publication duration excludes later
notification posts. Their jitter is substantially smaller than the observed
combined publication-interval jitter. This makes the earlier wait/start stage the
next measurement target; these aggregate populations do not yet measure its
delay or prove its contribution. The primary/combined summary `now` values are
only 433–525 us apart, with approximately matching retained coverage, but commands
are still unpaired. Do not subtract marginal variances or infer independent added
jitter. Combined observer receipt jitter rises above publication jitter toward
the end; observer wake timing can affect that population, which is separate from
producer publication timing. No simultaneous controller or camera records are
included here, and total loop latency remains unresolved. Primary and combined
publication tails still reach roughly -77/+80 us and -78/+78 us about 500 us.

Every record reports writer PID 2,989,283, `combiner_trigger_inode=0` and
`combiner_primary_counter=0`. Both `combinerPrimaryStart` and
`combinerPrimaryPublication` therefore remain unavailable with zero samples.
`unlinkedPrimary` increases by 28,029, exactly matching the increase in combined
events/receipts. This is missing upstream association, not a counted DM command
failure. Zero `invalidPrimary`/`repeatedPrimary` does not validate upstream trace
data: linkage is rejected before those checks.

Other quality totals remain unchanged throughout: primary duplicates/gaps
1,061/1,060; combined duplicates/gaps 1,030/353, invalid metadata 9 and invalid
trace 1. Missing traces, repeated starts, queue drops and rejections do not
increase. Primary receipts increase by 28,030 and combined receipts by 28,029;
the different observation endpoints do not establish a missing frame. Queue
occupancy is zero or one. Scheduler state remains 0, deliveries/misses remain
zero, and its single cumulative fault is unchanged. Enabled diagnostic flags
do not mean scheduling is enabled.

The operator was asked to inspect the FPS name and `.procinfo.triggermode`,
`.procinfo.triggersname`, `.procinfo.triggerdelay` for the identified writer,
without changing settings yet. Local MILK source confirms that a resolved named
primary supplies its inode and upstream trace even with DELAY mode 4 retained.
If the current configuration is unnamed/unresolved, naming the actual primary
(`dm01disp03` / `aol1_dmC`) and restarting dmcomb is the intended linkage step;
retain its existing wake policy/delay for that comparison. Naming the stream also
adds upstream trace-copy work, and the best-effort metadata/provenance limitations
still apply. An immediate switch to one-input semaphore triggering would alter
auxiliary-only update behavior and is not justified by this excerpt.

This updates only the debugging record and DM timing note. No code, test/build
rerun, commit, push, live configuration or agent instrument action is included.

### dmcomb trigger configuration confirmed

The operator confirmed `.procinfo.triggermode=4` (correcting an initial report of
5), a blank `.procinfo.triggersname`, and `.procinfo.triggerdelay=0.000010` seconds
(10 us). This matches the inspected DELAY mode and explains the zero trigger
inode. The delay is a requested sleep, not a measured primary-to-start delay or
an inferred jitter floor.

The next operator configuration step is to set only the trigger-stream name to
`dm01disp03`, retain mode 4 and the 10 us delay, and restart dmcomb through the
normal instrument procedure. The inspected dmcomb loads its numbered channel
streams before processinfo trigger initialization, so that channel name should
resolve when initialization records its inode. Keep synco scheduling disabled
for the baseline comparison. After restart, verify a nonzero matching trigger
inode and accumulating valid `combinerPrimaryStart`/`combinerPrimaryPublication`
samples in new telemetry. Their availability still depends on upstream trace
freshness and ordered timestamps. Naming the primary adds trace-copy work but
retains DELAY-mode all-channel scanning. No operator configuration change has
been reported as performed yet; the FPS instance name remains unspecified.

## 2026-10-23 20:22 UTC linked primary-to-combiner timing

The operator supplied seven records at
20:22:01.761752005–20:22:07.769232540 UTC, sequences 21,161–21,167, following
the requested trigger-name configuration/restart. No bintel filename or repeated
FPS mode/delay readback was included. The writer has changed to PID 2,053,564,
the trigger inode is consistently 1,525 and the named primary counter advances
from 200,294,486 to 200,306,501. Both linked populations are now valid with
3,999–4,000 observations per nominal two-second window. Primary and combiner
collector sessions remain 2 across the producer change; their summary times
are 533–543 us apart.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Primary publication interval | 499.999710–500.001234 | 3.633671–3.961510 | 433.790–574.866 |
| Synco primary receipt interval | 499.999703–500.001319 | 3.659141–3.991984 | 433.659–575.467 |
| Combined publication interval | 499.998548–500.002163 | 6.224304–6.475876 | 431.445–573.573 |
| Combined observer receipt interval | 499.998691–500.002108 | 6.314880–6.610741 | 431.445–573.894 |
| Primary trace update to combiner execution start | 6.506088–6.703962 | 3.413763–3.457118 | 0.551–12.734 |
| Combiner execution start to trace update | 4.857137–4.946693 | 0.148513–0.165380 | 4.338–6.081 |
| Combiner trace update to writetime | 0.314056–0.323214 | 0.020986–0.027013 | 0.290–0.832 |
| Primary trace update to combined writetime | 11.736252–11.896973 | 3.341926–3.387929 | 5.590–18.214 |

This locates the largest measured stage variation before combiner execution:
approximately 6.5–6.7 us mean with 3.4 us jitter, versus about 4.9 us execution
with 0.15–0.17 us jitter. That is consistent with variable polling phase under
the previously reported DELAY/10 us configuration, without proving a particular
wait distribution or actual iteration period. For each qualified population the
three stage means add to the primary-to-combined mean, within printed precision.
Stage variances must not be added/subtracted without covariance. Publication
intervals additionally contain differences between adjacent command delays and
upstream interval variation; their 6.2–6.5 us jitter is not the same statistic as
the approximately 3.4 us primary-to-combined delay jitter.

The upstream endpoint is the primary producer's trace-update timestamp, slightly
before its writetime. The output endpoint is combined writetime, before later
notification posts and controller work. Thus the approximately 11.8 us delay
does not include camera-to-primary, combined-to-controller wake or SDK latency.
Trace association remains best-effort metadata, not proven atomic pixel/counter
identity. The observed 18.214 us maximum in these short, overlapping windows is
not a qualified upper bound and does not relax any scheduler timing criterion.

Primary events/receipts, combined events/receipts and the named primary counter
all advance by 12,015 over the excerpt. Every problem counter is unchanged,
including `unlinkedPrimary=41,349,322` and `repeatedPrimary=1`. The scheduler
remains state 0 with zero deliveries/misses and an unchanged fault count of 2;
queue occupancy is zero or one. Compared with the 19:31 excerpt, primary gap and
scheduler fault totals have each increased by one, combined duplicate/gap totals
by 85/20 and repeated-primary by one. Those changes occurred somewhere between
the excerpts; their timing/cause cannot be assigned to the restart from these
records. No new anomaly is observed during this excerpt.

The next proposed single-variable comparison is a requested 5 us delay
(`.procinfo.triggerdelay=0.000005`), retaining mode 4, the named primary, placement,
priority and workload, with synco scheduling disabled. This increases polling
activity; compare CPU use as well as linked delay jitter/extrema, combined
cadence and quality counters over fresh windows. Local MILK source permits
runtime FPS delay updates. No such change has yet been reported as performed.
This evidence updates only the debugging record and DM timing note; no code,
test/build rerun, commit, push or agent instrument action is included.

### Primary-triggered comparison selected

The operator proposed switching to triggered operation on the current path. This
supersedes the proposed 5 us polling-delay experiment for the next baseline:
retain `dm01disp03` as `.procinfo.triggersname`, select semaphore
`.procinfo.triggermode=3`, and request automatic reader allocation with
`.procinfo.semindexrequested=-1`. Keep synco scheduling disabled and other
placement/priority/workload settings unchanged. The retained DELAY setting is
unused in mode 3. Obtain fresh linked-stage/combined/controller timing and check
actual mode/reader assignment; allocation failure can fall back to CNT0 mode 1.

Use a normal dmcomb restart for this transition. The inspected source allocates
and registers its semaphore reader during trigger initialization, whereas a live
FPS mode change alone only changes the mode field. Actual deployment and the
new comparison remain operator steps, not agent instrument actions.

Primary-triggered combination still includes all channel values on each real
primary wake, so it is compatible with the current disabled-synco baseline and
passive step-0 preparation (`steps=1`). Independent non-primary changes wait for
the next primary event. The present multi-step scheduler posts its designated
destination (normally `dm01disp07`) for later steps and never posts the primary;
those events would not wake dmcomb waiting only on `dm01disp03`. Before returning
to multi-step operation, a trigger route carrying both primary and designated
auxiliary events is required. This comparison does not implement that route or
change the existing scheduler's no-primary-post contract.

### Targeted primary-semaphore wake proposed for extra steps

The operator clarified the intended multi-step route: synco can post a semaphore
on `dm01disp03` for active extra steps while leaving the real primary command
unchanged. Local source review finds this workable as a tightly scoped exception
to the previous no-primary-post contract. Finish the existing passive batch,
publish the designated auxiliary destination counter, then issue one bare
`sem_post` to dmcomb's verified reader for each eligible `j>0` step. Step 0 adds
no wake. The auxiliary counter change remains necessary because dmcomb checks
channel-counter progression even after a semaphore wake.

Use only dmcomb's assigned reader, not primary fan-out or synco's own monitor
semaphore. The inspected `ImageStreamIO_sempost(image,index)` also updates
`semWritePID` and posts the logger semaphore, so it does not provide the strictly
targeted operation. Validate the stream identity, reader index/PID and actual
mode; never allocate/consume/flush/unregister dmcomb's reader on its behalf.
Reject stale assignments after restart. Primary pixels, metadata and other
reader/log semaphores remain untouched by the synthetic wake.

Retain final primary-counter/window checks, timestamp the actual wake and require
one combined-counter acknowledgement before another step. Recheck the primary
counter before accepting that acknowledgement: current `consume()` can accept
an existing output increment before reaching its primary-checking wait. Queued
real/synthetic wakes can coalesce, so post success is not proof of delivery. A
failed wake after auxiliary counter publication must disarm without rollback or
replay. Extra outputs intentionally reuse the primary trace identity; existing
linked-primary populations remain main-only and repeated-primary counts become
expected for those extras. The [main plan](dm-syncopation.md) records the proposal
and verification scope. No scheduler implementation or live change is included.

## 2026-10-23 20:33 UTC semaphore-triggered baseline

The operator confirmed dmcomb trigger mode 3 and supplied fifteen records from
`dmtweeter-synco_20261023191911531081671.bintel`, at
20:33:44.561096594–20:33:58.577849414 UTC, sequences 21,863–21,877. Both observer
sessions remain 2; the new stable writer PID is 2,070,021, trigger inode remains
1,525 and the named primary counter advances from 201,619,675 to 201,647,708.
Every interval population has 3,999 samples and every per-output stage has
4,000 valid samples per nominal two-second window. Summary times are 473–630 us
apart. The operator also reports that `dmtweeter` `[fgtimes]` jitter RMS has
roughly halved; no controller records were included to calculate that ratio here.

| Measurement | Window mean range (us) | Window sample-jitter range (us) | Reported event extrema (us) |
| --- | --- | --- | --- |
| Primary publication interval | 499.998599–500.001966 | 3.508605–4.009065 | 421.827–580.135 |
| Synco primary receipt interval | 499.998637–500.001949 | 3.526258–4.031853 | 422.087–579.986 |
| Combined publication interval | 499.999466–500.001177 | 3.534654–4.021923 | 421.617–580.296 |
| Combined observer receipt interval | 499.999366–500.001152 | 3.572648–4.060006 | 421.827–581.328 |
| Primary trace update to combiner execution start | 3.841527–3.874822 | 0.148693–0.153102 | 3.426–5.811 |
| Combiner execution start to trace update | 4.918682–4.943921 | 0.071946–0.082437 | 4.558–5.801 |
| Combiner trace update to writetime | 0.561251–0.566026 | 0.042448–0.045912 | 0.400–1.062 |
| Primary trace update to combined writetime | 9.323245–9.379611 | 0.164774–0.174293 | 8.756–11.282 |

Compared with the mode-4 20:22 excerpt, primary-to-start delay jitter falls from
3.414–3.457 us to 0.149–0.153 us; total trace-primary-to-combined delay jitter
falls from 3.342–3.388 us to 0.165–0.174 us, approximately twentyfold. Mean total
delay falls from 11.736–11.897 us to 9.323–9.380 us. Combined publication interval
jitter falls from 6.224–6.476 us to 3.535–4.022 us and now closely follows the
primary's 3.509–4.009 us. This strongly supports polling as the source of most
of the previously observed additional combiner-path timing spread. The reported
controller improvement is consistent with that result. These are descriptive
comparisons across separate, overlapping-window excerpts; do not subtract
marginal variances or claim a pooled independent jitter estimate.

Primary/combined events and receipts, and the named primary counter, each advance
by 28,033. All problem counters remain unchanged throughout, including combined
`unlinkedPrimary=41,349,322` and `repeatedPrimary=13`. Primary duplicates/gaps are
1,061/1,062; combined duplicates/gaps are 1,144/379. The ordinary valid scheduler
snapshots remain state 0, two requested steps, zero deliveries/misses and three
cumulative faults. The 20:33:55 record has `schedule_valid=0`: its zero steps,
period and faults are unavailable-snapshot placeholders, not a scheduler reset.
Both diagnostic tables remain valid in that record. Queue occupancy is zero or
one. Since the last 20:22 record, primary gaps and scheduler faults each rose by
one, combined duplicates/gaps by 29/6 and repeated-primary by 12. These increments
occurred in the unobserved interval/changeover and cannot be assigned to steady
mode-3 operation from this evidence.

The remaining cadence variation and roughly -78/+80 us primary interval tails
are already upstream of dmcomb. The much steadier combiner delay does not by
itself qualify synco's primary-interval bounds or enabled extra steps. Existing
trace-clock, best-effort association and endpoint limitations remain: these
durations do not include camera-to-primary or combined-to-controller/SDK work,
and the observed maximum is not a qualified bound. The result supports the
targeted primary-reader wake proposal, which remains unimplemented and requires
its own extra-step timing/delivery verification. Only engineering records are
updated; no code, test/build rerun, commit, push or agent instrument action is
included.

## 2026-10-23 direction: median cadence and skipped auxiliary cycles

The operator asked to consolidate the current work and accept the observed
large interval excursions as skipped frames where possible, with syncopation
based on the median interval. The completed work is timing instrumentation and
the operator's switch of dmcomb to mode 3, which removed most additional
combiner delay jitter. The targeted extra-step wake and new median/skip policy
are still design work, not C++ changes.

Inspection shows why the present app repeatedly reacquires: even though period
selection is independent of a single interval, `processPrimary()` still faults
on each interval outside selected-period +/- `earlyMainUs` (20 us by default).
That clears output/phase priming and learned corrections, then requires 32 clean
frames. Replacing only the period estimator would leave that behavior intact.

The revised [main plan](dm-syncopation.md) uses a rolling primary-interval
median, includes ordinary tails in that estimator, and keeps every real primary
as the phase anchor. Duplicates and gap-spanning/invalid samples are excluded.
Camera FPS may seed acquisition and remains diagnostic; it no longer overrides
a valid median. Ordinary excursions cancel obsolete auxiliary targets and count
as skips while retaining the median/feedback. Resume requires known current
output and fresh on-time passive preparation, not another run of 32 near-median
intervals. Forward counter gaps require a bounded baseline resynchronization;
resets, resource changes, ambiguous output and actual unsafe copy/read overlap
still stop operation. No skipped main or auxiliary trigger is synthesized or
replayed.

The early-arrival reserve remains a separate engineering choice. An early main
cannot be prevented from overlapping an already-issued command merely by
skipping after its arrival. The operator was asked about a provisional 100 us
reserve against the observed approximately 80 us early intervals. With the
current delivery/keepout/guard/copy/observation/visibility/reserve defaults, this
would require 273 us per half-cycle in the existing feasibility test, exceeding
the 250 us available for two steps at 2 kHz. The operator explicitly chose to
leave this reserve undecided, so 100 us is not selected. The current 20 us
setting remains unchanged and unqualified for the larger excursions. No timing
budget has been relaxed or instrument setting changed. This is a plan/evidence
update only, with no code, test/build rerun, commit or push.

### Can present telemetry establish excursion ordering?

The operator asked whether early arrivals follow late arrivals, or vice versa.
Current saved `[dmsynco]`/`[dmtimes]` timing tables contain count, mean, sample
deviation, extrema and coverage, not ordered interval pairs or raw timestamps.
Overlapping windows do not recover immediate adjacency or a directional
correlation. The INDI fault publication/observation values describe the same
latched interval in different clocks; they are not neighboring intervals.

For a 500 us nominal period, an isolated delayed timestamp can produce intervals
580 then 420 us, while an isolated advanced timestamp can produce 420 then
580 us. Both pairs have identical aggregate statistics. A short interval is early
relative to the previous observed timestamp plus the period; it need not be
early relative to the camera/AO timing grid. Negative lag-one interval
correlation alone would not distinguish these two cases.

Ordered primary publication events already exist in the collector's bounded
consumer window. A possible addition is consumer-side long-to-short versus
short-to-long transition counts and conditional rates, next-interval residuals,
and two-interval sum residuals. Alternatively, export a bounded ordered
timestamp/counter trace. Break associations at duplicate/gapped counters, loss,
resets or invalid timestamps and keep publication/receipt clocks separate. This
is a measurement proposal only; no new fields, code or instrument actions are
included, and the early-arrival reserve remains undecided.

### Ordered interval-pair check implemented

The operator requested building the ordering check, expecting late-then-early
to dominate. The implementation tests both directions without that assumption:
the primary collector uses existing publication observations to compute the
window median and pair statistics on its ordinary consumer thread. No producer
event/timestamp capture, scheduler period, recovery policy or trigger route is
changed. Other users of the shared collector leave this optional analysis off.

The new diagnostic-only `diagnostics.excursionUs` defaults to 20 us and zero
disables pair analysis without disabling the existing collectors. Eligible
three-event sequences must be real, contiguous, timestamp-valid and within the
window. Duplicates/gaps/invalid events break the incoming chain. Next-interval
populations after late/early intervals include ordinary successors; directed
opposite-excursion sum counts form the corresponding numerators. Thus a displayed
`lateEarly:N/D` means N early successors among D eligible late follow-ups, not a
count divided by all late intervals including those without an observed next
event. Reverse evidence is retained independently. Pair sums can be compared
with twice the reported diagnostic median.

The app adds read-only `primary_interval_pairs` and an optional appended primary
table in `[dmsynco]`. The text formatter reports both fractions and four complete
populations. Older buffers remain readable and explicitly report pair data
unavailable; new disabled summaries remain distinct. The
[guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents seconds versus
the microsecond configuration, sparse data and overlapping windows. These
interval-order measurements do not establish absolute camera phase or prove
that every future excursion follows the same pattern.

Normal library/schema, app and logdump builds pass. Focused workstation suites
pass **4,540 assertions / 73 cases**: collector 317/15, app 2,748/36,
new pair/controller logger 1,000/7, legacy synco logger 322/7 and default-disabled
collector integration 153/8. Formatting/whitespace and independent review pass.
The test fixture's conditional count and legacy formatter expectation were
corrected during verification; the final suites all pass. No extra RT timestamp
capture or producer event changes were introduced. No hardware qualification,
agent instrument action or commit/push is included. The new
scheduler behavior and targeted extra-step wake remain pending, with the early
reserve undecided.

## 2026-10-23 operator result: late-then-early predominates

The operator supplied 15 new `[dmsynco]` records, sequences **26–40** in primary
session **2**, logged from **21:32:58.957127125** through **21:33:12.977516466 UTC**.
The source telemetry file is
`dmtweeter-synco_20261023213233925248850.bintel`, provided in attachment
`f68a84e8-2846-46e4-837a-7bff40fd55d9/Pasted text.txt`. The primary-pair table is
present and enabled, with a **20 us** threshold and about **3,999 intervals /
3,998 adjacent pairs** per two-second window. All reports have valid primary
timing and a captured schedule, `state=0`, `steps=2`, and zero delivered, missed
and fault counters. These records measure ordering without demonstrating active
extra-step delivery; the enable toggle is not present in this telemetry.

Reports arrive about once per second, so neighboring two-second windows overlap.
For a sample without shared observations, select sequences **26, 28, 30, 32, 34,
36, 38, 40**. Their `primary_now_ns` times are at least **2.002860006 s** apart,
leaving 2.860–2.978 ms between nominal window boundaries. These eight windows
contain **31,992 eligible intervals / 31,984 adjacent pairs**; this is a sampled
set of disjoint windows, not a cumulative total over the continuous run.

| Directed follow-up | Count | Conditional fraction |
| --- | --- | --- |
| Late interval immediately followed by early | 49 / 65 eligible late follow-ups | 75.38% |
| Early interval immediately followed by late | 6 / 69 eligible early follow-ups | 8.70% |

The alternate disjoint selection, sequences **27, 29, 31, 33, 35, 37, 39**, gives
**46/59 (77.97%)** and **5/64 (7.81%)**, respectively. It supports the same
directional preference but overlaps the first selection, so do not combine the
two selections or treat them as independent replicates. Excursions can cluster;
these fractions are descriptive observations, not an independent-event
probability model.

Across the first disjoint selection, the next interval after a late interval
averages **472.591 us** (65 samples), whereas the next interval after an early
interval averages **505.258 us** (69 samples). The 49 late-then-early sums average
**1,000.678 us**, close to twice the roughly 500 us median, supporting substantial
two-interval catch-up. Individual sums span **951.037–1,038.631 us**, however, so
the next interval does not exactly repay each delay. Sixteen eligible late
intervals have no immediate successor classified early at this threshold.

This supports the operator's expectation as the dominant pattern, not as an
absolute rule. Nonzero early-then-late counts also do not prove an independently
early initiating event: a late–early–late sequence contributes both directed
pairs. The saved pair summaries cannot distinguish such rebounds from separate
early-first disturbances. The fractions condition on the first interval and
its observed successor, not directly on whether every short interval has a late
predecessor. Classification still measures interval length, not absolute phase
against the camera grid.

The median across all 15 windows is **499.944–500.145 us**; primary publication
interval deviation is **5.687–6.252 us**, with overall extrema
**419.221–580.557 us**. Primary publication and receipt statistics closely agree.
The combiner remains writer PID **2070021**, trigger inode **1525**. Its
primary-to-combined-publication mean is **9.305–9.334 us**, with deviation
**0.171–0.179 us** and observed maximum **11.151 us**. The much smaller variation
in that delay remains consistent with most output cadence variation already
being present at primary publication.

Primary event totals increase **48,462 -> 76,503** (+28,041) between first and
last records. All reported primary/combiner quality-error totals remain zero,
including duplicates, gaps, rejected/invalid metadata, missing/unlinked trace
and collection drops. Queues contain at most one observation. This supports
using the reported contiguous-event populations for this comparison; it is not
a proof about unobserved instrument paths or physical DM response.

Record the dominant compensation pattern in the main plan, but do not make
exact catch-up or a universal late-first order a scheduler assumption. The
median/skip revision and targeted wake remain pending, and these measurements
do not choose or qualify the early-arrival reserve. Only engineering records
were changed for this analysis; no C++ changes, test/build rerun, commit, push
or agent instrument action was performed.

## 2026-10-23 clarification: keep active steps on the predicted cadence

The operator clarifies the intended ordinary-excursion handling: an early main
can skip passive step-0 preparation while the following active step 1 remains
on the regular cadence; a late main should not move step 1. The main plan is
revised to distinguish the predicted median-period target grid from actual
main arrival and from passive preparation success. This supersedes the prior
proposal to re-anchor every active target to each observed main and to require
fresh successful step-0 preparation before resuming active work.

For a predicted main at **500 us**, `T=500 us` and two steps, the active step-1
target stays **750 us** for either a main at **420 us** or one at **580 us**.
An early main cancels a passive batch that has not started and cannot safely
precede it; if the batch is already complete there is no preparation to skip.
A skipped preparation leaves the last complete contribution in place and is
counted without a routine primary-fault/reacquisition response. The next active
step independently stages/copies its contribution and does not depend on that
passive preparation having succeeded.

For a late main, an already prepared passive step 0 is still ready on time;
its main-triggered application to the DM is late. The active target does not
slide, but execution still requires association of the actual main's combined
output and expiration of its reader/processing hold. A main arriving too close
to the target, or not yet observed/associated, makes that active step unusable:
skip it rather than postponing it or synthesizing catch-up triggers. A main
arriving during an in-progress passive copy is a reader/partial-batch fault,
not an ordinary silent skip. Skipping future preparation also cannot undo a
collision with a previously issued active command whose hold is still running.

Source inspection confirms the current code has both superseded couplings:
`processPrimary()` targets `observed + j*T/Nreq` and gates all active steps on
`m_primed`, which comes from successful passive preparation. It also faults on
the narrow primary-interval residual. The proposed behavior therefore requires
scheduler changes rather than a configuration adjustment. Median period alone
does not specify phase; initial grid acquisition, persistent phase correction
and ambiguous-cycle handling remain design details to settle. The next-main
envelope must allow catch-up after a late main rather than assuming another
full `T` from that late arrival. The early-arrival reserve remains undecided.

Only the main plan and this debugging record were revised. No C++ changes,
test/build rerun, commit, push or agent instrument action was performed.

## 2026-10-23 upstream stage attribution: current coverage and next diagnostic

The operator asks whether existing telemetry can identify which AO processing
stage produces each excursion, predicting steady `camwfs.atime` followed by a
delay in one or more later stages. Current measurements show the large interval
excursions already at `dm01disp03` publication, while the latest measured
primary-to-combined delay has only 0.171–0.179 us deviation. They do not locate
the excursion among camera publication, WFS preparation, reconstruction, filter
and DM-command production. Separate camera/primary aggregate windows cannot
associate a particular delayed command with its camera acquisition or determine
whether the responsible stage is repeatable.

Local source inspection confirms `dev::dmCombTiming::observe()` captures only
the combined trace's first two rows, representing dmcomb and the primary
producer. It publishes aggregate combiner durations, not a frame-associated
upstream timing chain. MILK already propagates additional process-trace rows
containing writer PID, trigger inode/input counter, process-start and
stream-update times. A possible next diagnostic is to copy a bounded upstream
chain at primary observation and calculate each stage's start-to-update work
and upstream-update-to-start handoff. Condition those measurements on the same
late/early primary event, retaining its preceding/following observations; compare
latency changes between successive associated frames, since a constant long
stage latency does not by itself create interval jitter. Preserve all stage
contributions rather than forcing every excursion into one cause.

This could reuse existing producer timestamps without adding AO hot-path clock
reads, but trace fields are best-effort metadata: input counters are sampled at
output update, not proven consumed-frame identity. Validate chain continuity,
counter/frame association, missing/frozen rows and writer/session changes before
claiming event attribution. `camwfs.atime` is not carried by the process trace;
matching a specific DM excursion to its EDT acquisition timestamp would need
a separately matched camera frame. The
camera frame grabber's REALTIME publication timestamp, MILK external trace
clock and local monotonic markers also need qualified clock relationships before
cross-clock latency subtraction. Ordinary aggregate stage statistics alone
would not establish which stage caused a given excursion.

This records a diagnostic proposal, not a new implementation request. Only
this engineering record was changed; no C++ changes, tests, build, commit, push
or agent instrument action was performed.

### Independent camera acquisition excursion check

The operator clarifies that camera acquisition can be analyzed separately to
test whether `camwfs.atime` ever has interval excursions. This does not require
camera-to-DM frame matching or subtraction of clocks from separate systems.
Analyze consecutive acquisition differences in their own clock over the same
test period, recording window coverage, extrema and any threshold-exceeding
intervals. Existing `camwfs.fg_timing.acq_min`/`acq_max` already report these
extrema for each analyzed window; mean/RMS alone cannot establish absence of
rare excursions. The current `[fgtimes]` log stores means/jitters without these
extrema. Continuous coverage matters when claiming none occurred.

If covered camera intervals remain steady while primary intervals have large
excursions, that places the added variation after camera acquisition. The
propagated AO trace can then locate variation in later handoff/processing
stages. Exact event association remains relevant only for correlating a specific
camera event with a specific command or calculating full camera-to-DM latency;
it is not a prerequisite for this independent camera check. No implementation
or hardware action is implied by this clarification.

## 2026-10-23 independent camera and upstream checks implemented

The operator authorized implementing the two diagnostic checks. A third
metadata-only monitor independently observes camera acquisition `atime`, using
existing `dev::dmTiming` interval/median/pair analysis. It defaults to `camwfs`
and ordinary scheduling, with automatic reader allocation. Acquisition extrema,
counts, coverage, quality and excursion ordering are retained in `[dmsynco]` and
read-only INDI rather than depending on mean/RMS-only camera `[fgtimes]` logs.
No camera-to-DM frame pairing is attempted. Malformed and nonincreasing stamps
at consecutive counters are rejected; gaps/losses prevent a claim of complete
camera coverage. Camera pixel data and producer settings are never changed.

The new reusable `dev::dmPipelineTiming` captures up to ten existing primary
trace rows before the cycle mutex, reusing existing primary receipt/entry times.
Its producer performs bounded metadata copies without allocating, waiting or
reading another clock. Ordinary app logic analyzes each stage's processing and
handoff duration, plus changes from the previous associated observation,
conditioned on late/early primary publication intervals around a retained-window
median. Signed changes separate newly added delay from constant stage latency.
Oldest represented start intervals and primary publication-tail changes retain
variation outside the represented stages. The camera acquisition endpoint is
independent of that oldest trace boundary.

Qualification requires real consecutive primary events and fresh coherent rows
with independently progressing trigger-input counters, starts and updates.
Identity changes break associations and statistics include only the latest
observed topology. Existing MILK MeasureTiming must keep starts advancing; no
producer timing, trigger or scheduling configuration is changed. Missing,
invalid/frozen, truncated, changed and lost observations are reported explicitly.
Trace metadata still does not prove exact consumed-frame identity. All external
clock differences remain separate from monotonic observation coverage.

New optional `[dmsynco]` camera/pipeline tables append to the existing schema;
legacy buffers remain readable. Camera interval properties reuse primary
cadence/quality/pair registration and update code. New pipeline properties show
row identity, ordinary and conditional populations, signed changes, boundary
intervals and quality. Both checks continue while synco scheduling is disabled,
auxiliary resources are unavailable or a timed cycle holds the scheduler mutex.
`diagnostics.camera` and `diagnostics.pipeline` default true; `cameraMonitor`
selects the independent acquisition stream. Window/cap/threshold settings are
shared with existing diagnostics; a zero excursion threshold keeps ordinary
trace measurements while disabling conditional/pair classification.

The guide documents metric names, seconds versus configured microseconds,
observation coverage, row indices versus synco step numbers, clocks and source
boundary limits. Main-plan scheduler revisions remain pending: this adds no
median-grid scheduling, per-step skip policy or targeted extra-step wake, and
the early-arrival reserve stays undecided.

During local verification, the new configuration regression caught the monitor
helper overwriting the camera name with the app instance. Restoring `camwfs`
after helper setup fixes the intended default. The new logger test also needed
its explicit generated API include; that test-only build issue was corrected.
Final verification results and the requested commit/push checkpoint follow.

### Local verification and requested commit/push checkpoint

Final focused workstation suites pass **4,673 assertions / 64 cases**:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| App and isolated primary/combined/camera semaphore workers | 2,897 | 41 |
| Upstream stage collector | 99 | 5 |
| New acquisition/pipeline telemetry | 355 | 4 |
| Legacy synco telemetry | 322 | 7 |
| Existing ordered-pair/controller telemetry | 1,000 | 7 |

Normal library/schema, dmSyncopation and logdump builds pass. No new C++ warnings
were emitted; the library's existing schema field-name notices and dirty-tree
build notices remain unrelated. All touched C++ files pass clang-format checks,
whitespace checks pass, and independent collector/serialization/app/guide reviews
found no remaining blockers. The final app run includes the fixed camera default.
Local tests used isolated temporary shared memory only; test logs were preserved
outside the repository. No instrument commands or hardware actions were run.

The user authorized committing and pushing after verification. Keep functional
code/tests and these associated engineering records together, followed by the
user guide as a documentation-only commit. Operator deployment must rebuild and
install both dmSyncopation and logdump, then compare camera extrema and upstream
conditional changes while scheduling remains disabled. That instrument result,
not the workstation checks, will determine whether acquisition or a particular
handoff/processing stage carries the observed excursions. Scheduler revisions
and selection of the early-arrival reserve remain separate pending work.

## 2026-10-23 first independent camera and upstream-stage operator run

The operator supplied twelve `[dmsynco]` records, sequences **12–23**, from
**22:12:01.890510637** through **22:12:12.930420077 UTC**, in
`dmtweeter-synco_20261023221150854513305.bintel` (attachment
`3f4455e3-6f2f-440f-89fe-ec7dfcd3aaf2/Pasted text.txt`). The new camera and
pipeline tables are present/enabled, session **2**, two-second windows with
**3,999** qualified intervals each. Every schedule snapshot is valid with
`state=0`, `steps=2`, and zero delivered/missed/fault counters; this measures the
ordinary AO path without demonstrating active extra-step delivery.

| Observed interval | Overall extrema, us | Window deviation range, us |
| --- | ---: | ---: |
| Camera acquisition (`atime`) | 480–525 | 0.325–0.819 |
| Camera observer receipts | 422.718–579.725 | 1.927–3.453 |
| Oldest represented AO stage starts | 422.198–580.737 | 1.465–3.206 |
| Primary command publication | 421.687–582.079 | 5.036–5.705 |

Camera acquisition is much steadier than primary publication, but is not
excursion-free: sequences **14/15** report a **525 us** maximum and one eligible
late follow-up at the strict **20 us** threshold around the 500 us camera
median. Those windows overlap, so their counts must not be added. A 480 us
interval is exactly on the early boundary and is ordinary under the strict
classification. Camera observer receipts have much larger tails than the
acquisition timestamps, but include the observer's own scheduling; they do not
measure the camera publication interval.

The four stable trace rows are:

| Row | Writer PID | Trigger-input inode | Previously identified path role |
| --- | ---: | ---: | --- |
| 0 | 2035667 | 1614 | DM-command conversion, input `aol1_modevalDM` |
| 1 | 2035467 | 1600 | Filtering, input `aol1_modevalWFS` |
| 2 | 2035310 | 1589 | Reconstruction, input `aol1_imWFS2` |
| 3 | 2035165 | 2411 | `acquWFS` preparation, input `camwfs` |

Names/roles reuse the earlier operator trace mapping; this record directly
reports PID and input inode, not process names or output-stream identity. The
oldest row starts after camera publication/wait; it does not timestamp camera
acquisition itself.

Select sequences **12, 14, 16, 18, 20, 22** for six disjoint two-second pipeline
windows. These contain **46 late / 49 early** primary intervals at the 20 us
threshold. The same every-other selection also has disjoint camera windows,
with **23,994** acquisition intervals and **one eligible late follow-up / zero
early follow-ups**. Counts are from sampled windows with small gaps, not an exact
cumulative event count over the entire continuous capture. No late-to-early
camera pair is counted in that selection.

| Same pipeline cohort | Late primary intervals | Early primary intervals |
| --- | ---: | ---: |
| Qualified observations | 46 | 49 |
| Oldest-stage start interval, weighted mean, us | 538.737 | 461.926 |
| Sum of processing/handoff changes and publication-tail change, us | -1.191 | +1.457 |
| Primary interval reconstructed from those components, us | 537.546 | 463.383 |

The last two rows are derived from like-count conditional populations, including
signed changes across all four stages, the three represented handoffs and the
primary publication tail. They are not subtraction/addition of marginal jitter
or camera/primary clocks. The decomposition shows the dominant late/early
variation already at the oldest represented start. Subsequent measured work
partly compensates it on average. Later-stage execution/handoff changes do not
add tens of microseconds to these large excursions. Ordinary processing still
contributes smaller cadence variation; the first-stage execution itself ranges
62.508–79.230 us across all observations, so this does not claim that downstream
processing has zero jitter.

Combiner primary-to-publication mean remains **9.302–9.337 us**, with deviation
**0.163–0.186 us**. Camera, primary, combined and pipeline quality-error counters
all remain zero throughout: no rejections, duplicates, gaps, invalid metadata,
missing/frozen trace, topology changes, truncation or queue drops are reported.
Each row's stage population has 4,000 qualified observations per window. This
supports the internal comparison while retaining the documented best-effort
trace/frame-association limits.

The result narrows the large-tail investigation to **camera publication and the
camera-to-first-AO-stage wake/scheduling region**. It does not identify a single
cause inside that region. The independent acquisition series has a smaller
rare excursion, and cannot associate that particular event with a DM event.
The next useful distinction is independent camera `writetime` intervals versus
oldest-stage start intervals; the existing camera observer receipt population
cannot replace that measurement. No new implementation or producer setting
change is made in response to this attachment. Only the main plan and this
record were updated; no test/build rerun, commit, push or agent instrument action
was performed.

### Verified OCAM acquisition timestamp source

At the operator's request, source inspection verifies that `camwfs.atime` uses
the EDT PDV SDK-provided timestamp in the OCAM controller path:

1. `ocam2KCtrl::acquireAndCheckValid()` passes `m_currImageTimestamp` by reference
   to `edtCamera<ocam2KCtrl>::pdvAcquire()`.
2. That helper calls `pdv_wait_last_image_timed(m_pdv, dmaTimeStamp)` and copies
   the returned two values directly to `tv_sec` and `tv_nsec`, respectively.
3. `frameGrabber` assigns that timestamp to the primary stream's `md->atime`,
   its circular timestamp array when enabled, and its acquisition statistics.

OCAM pixel descrambling/binning does not replace this timestamp. The post-publish
hook copies it unchanged to the synchronization stream without modifying the
primary. Publication `writetime` is separately sampled with `CLOCK_REALTIME`.
This verifies the source assignment path, not the exact SDK/driver timestamp
event or its clock implementation. No code or instrument action was required.

## 2026-10-23 isolated camera publication and acquWFS setup implemented

The operator requests measuring camera publication and `acquWFS-1` starts with
only those two AO data-producing stages running. Add a separate camera
`writetime` collector beside the acquisition collector, using the same coherent
metadata snapshot and existing callback clock readings. Timestamp histories and
statistics remain independent; malformed/nonprogressing acquisition metadata
does not discard valid publication metadata, or vice versa. Reuse `dev::dmTiming`
for interval extrema, ordered excursions, coverage and quality. New INDI
properties use the `camera_write` prefix; the appended optional telemetry table
prints `cameraWriteInterval`, `cameraWriteReceiptInterval`, `cameraWrite_pairs`
and the conditional successor/sum populations. The matching app and reader must
both be installed.

New `diagnostics.only=true` permits an arbitrary observational primary, skips
all scheduler DM-resource attachments and cycle work, creates no auxiliary or
schedule streams, forces combined monitoring off, selects an available primary
reader automatically, and rejects scheduler enable requests. Its schedule
snapshot is unavailable while independent measurements remain active.

The new [profile](../../../apps/dmSyncopation/camera-acquWFS.conf) selects
`aol1_imWFS2` and ordinary observer scheduling, clears the primary observer
cpuset, and disables the camera INDI FPS subscription. The new
[procedure](../../../apps/dmSyncopation/doc/camera-acquWFS-test.md) uses the
verified FPS RUN controls to stop downstream conversion/filter/reconstruction,
retains acquisition and CONF/settings, starts a uniquely named observer,
captures at least two minutes, and restores previously running stages with the
loop open. A one-row trace identifies `acquWFS-1` and its camera input inode.
The source interval measures processing starts after the trigger wait, serving
as a wake-cadence proxy. It does not timestamp exact semaphore return. Independent
acquisition, publication and source-start interval populations locate where
excursions first appear statistically; matching extrema or absolute cross-clock
latency are not inferred. Normally overlapping two-second windows require
checking actual record times and avoiding double-counted excursion totals.

Local verification passed **4883 assertions in 69 test cases**:

- Full `dmSyncopation` suite: 2972 assertions, 44 cases, including the actual
  isolated profile, independent acquisition/publication validity and session
  resets, and diagnostic-only resource/cycle bypass with rejected enable.
- `dmPipelineTiming`: 186 assertions, six cases, including a single camera-triggered
  row with separate wake, execution and publication-tail excursions.
- `dmPipelineTelemetry`: 403 assertions, five cases, including distinct acquisition
  and publication populations, JSON/text, old records and malformed new tables.
- Existing `dmSyncoTelemetry` and `dmTimingTelemetry`: 322 and 1000 assertions,
  respectively, seven cases each.

Normal library/schema, app and `logdump` builds passed. Clang-format validation
of all touched C++ files and the whitespace check passed. The first full-suite
attempt exposed an assertion that incorrectly assumed no earlier test had left
scheduler files; the new resource-bypass case now uses its own unused names and
the complete suite passes. Workstation-local test log artifacts were moved to
`/tmp/synco-isolated-test-logs-nt6im2d8`; build/test transcripts use the
`/tmp/synco-isolated-` prefix. No hardware actions were performed by an agent.
Scheduler revisions remain unimplemented and the early-arrival reserve remains
undecided.

## 2026-10-23 first camera publication capture: acquWFS comparison missing

Source: operator attachment
`ac20761e-b338-4348-b0b9-5857fe10cafc/Pasted text.txt`, beginning with
`teldump -f dmtweeter-synco`. It contains 17 records, sequences 40–56, at
2026-10-24 03:42:56.707644279–03:43:12.743490974 UTC (2026-10-23 evening
locally). Binary source name:
`dmtweeter-synco_20261024034217622125816.bintel`. The new publication table is
present, establishing that this app/reader pair supports it; the exact deployed
Git hash is not included in this capture.

| Population | Window mean interval, us | Window deviation, us | Observed extrema, us |
| --- | --- | --- | --- |
| SDK acquisition timestamps | 500.000250–500.000500 | 0.304–0.967 | 477–525 |
| Camera publication timestamps | 499.981280–500.000712 | 2.314–3.739 | 424.031–578.262 |

Every camera population contains 3999 valid intervals in the retained two-second
window, with near-full spans and sub-millisecond ages. Both acquisition and
publication collectors report zero rejected, duplicate, counter-gap, invalid and
dropped observations in every record. Counts progress from 76369 to 108441 for
acquisition and from 76371 to 108442 for publication. The slight count offset is
consistent with independent consumer updates and does not represent extra
camera frames. Acquisition has rare -23/+25 us interval excursions; publication
has approximately -76/+78 us excursions around the 500 us cadence.

The large interval tails are already visible at camera publication, rather than
requiring a later reconstruction/filter/DM stage to introduce them. The
independent timestamp populations do not supply matched frame delays or an
absolute acquisition-to-publication duration, and do not identify which camera
processing, timestamp or scheduling component causes the variation.

Nine separated windows (sequences 40, 42, ..., 56) have publication directed-pair
totals of **47/77 late-then-early** and **48/117 early-then-late**, using each
window's own median and 20 us threshold. These are qualified intervals with an
eligible successor, not all late/early events. Normal adjacent records overlap
and must not be added again. The ordering is not universal and the two
denominators differ; these totals do not prove causality or exact compensation.

The intended acquWFS series is absent. Primary events/receipts remain zero with
`primary_valid=0` and age -1; pipeline events/receipts/intervals and stage count
also remain zero. This is lack of observations, not a quiet wakeup population.
`schedule_valid=1` and `combiner_enabled=1` are inconsistent with an active
`diagnostics.only` profile. The normal instance is being read, but its primary
stream configuration and actual running producer set cannot be determined from
these records. No claim is made that `acquWFS-1` was stopped or that its wake
timing was steady. Camera receipt intervals describe the independent diagnostic
monitor's scheduling, not acquWFS wakeup.

Next, use the standalone `wfs-acqu-timing` observer with the shipped profile
while the operator retains only camera/acquWFS AO producers. Confirm
`schedule_valid=0`, `combiner_enabled=0`, advancing primary receipts and
`pipeline_stages=1` with the expected PID/camera input inode before a two-minute
capture. The procedure now explicitly lists these mode checks. No functional
code, scheduler policy or timing bounds were changed. Only engineering records
and the operator procedure were updated; whitespace checks passed. No build/test
rerun, commit, push or agent instrument action was performed for this analysis.

## 2026-10-23 standalone observer startup: required named configuration

The operator attempted the documented standalone command with
`-n wfs-acqu-timing -c "$PWD/camera-acquWFS.conf"`. Startup failed because
`/opt/MagAOX/config/wfs-acqu-timing.conf` did not exist; the observer did not
start and this failure supplies no additional timing data.

The launch procedure omitted a mandatory configuration prerequisite. Source
inspection of mxlib `application::setup()` confirms that it reads the required
local/instance file and returns on failure before parsing the `-c` option.
`MagAOXApp::setDefaults()` constructs that required path from the instance name.
The earlier review correctly identified the override order but did not account
for the missing-file failure before the override; the profile parsing unit test
also did not exercise this full framework startup path.

Corrected operator commands, from `apps/dmSyncopation`:

```bash
sudo install -m 0644 camera-acquWFS.conf /opt/MagAOX/config/wfs-acqu-timing.conf
/opt/MagAOX/bin/dmSyncopation -n wfs-acqu-timing
```

The procedure and main plan now include the prerequisite. This installs the
isolated profile under its separate instance name, without changing the normal
synco configuration or the framework's global configuration directory. No C++
change or rebuild is needed for this correction. Source/command review and
whitespace checks are the applicable verification; no tests were rerun. No
agent instrument action, commit or push was performed.

## 2026-10-23 isolated camera/acquWFS capture: excursions already at processing start

Source: `/home/jrmales/wfs-acqu-timing.txt`, 85497 bytes, SHA-256
`5c78ecdf97e4e73974a8593bfee751748f9e59140e49904dd71d3b81a5e79d4a`.
It contains 13 records, sequences 50–62, at 2026-10-24
03:55:55.479943403–03:56:07.514082595 UTC, or October 23
20:55:55–20:56:07 in Phoenix. This is a roughly 12-second excerpt, rather than
the full two-minute comparison requested in the procedure. Parsed populations
are retained locally at `/tmp/synco-isolated-20261024T0355.json`.

The profile is active: all records show `schedule_valid=0`,
`combiner_enabled=0`, valid primary/camera populations and one stable trace row
(`row=0`, PID 2263601, trigger input inode 2411). There are no topology changes,
missing/invalid traces, rejected metadata or collector queue drops. Trace
identity is consistent with the intended camera-triggered acquWFS observation;
the capture does not independently enumerate all RUN processes.

| Interval population | Window deviation, us | Overall observed extrema, us |
| --- | --- | --- |
| Camera SDK acquisition | 0.646–0.906 | 480–520 |
| Camera publication | 2.810–3.510 | 437.045–579.825 |
| acquWFS processing start | 2.164–3.288 | 430.132–577.219 |
| acquWFS output publication | 2.585–3.585 | 430.012–577.320 |

All means remain approximately 500 us. Camera populations each contain 3999
intervals per two-second window, with zero duplicate, counter-gap, rejected,
invalid or dropped observations. Camera publication has substantially larger
tails than SDK acquisition. Those large tails therefore exist before later AO
reconstruction/filter/DM processing is needed to produce them. These are
independent interval populations, so their extrema cannot be matched by frame
or their variances subtracted to estimate added wake jitter.

acquWFS execution averages 62.6546–62.6945 us, with 0.966–1.092 us window
deviation and extrema 61.676–76.154 us. Seven separated windows (sequences 50,
52, ..., 62) supply internally matched conditional decomposition:

| Output cohort | Count | Source-start interval, us | Execution change, us | Publication-tail change, us | Reconstructed output interval, us |
| --- | --- | --- | --- | --- | --- |
| Late | 53 | 544.1725 | +0.0683 | +0.00157 | 544.2424 |
| Early | 68 | 461.7888 | -0.1095 | +0.00153 | 461.6808 |

The values are weighted by each qualified cohort's count. Most output interval
variation is already present at processing start; changes in execution and
output-publication tail contribute much less. The trace has no represented
upstream camera stage, so `stageHandoff` is unavailable and matched camera
`writetime` to acquWFS-start latency is not collected. The remaining region
includes camera acquisition-return/processing/publication and acquWFS wake or
trigger handling. The new data does not uniquely assign each excursion within
that region.

Primary/pipeline observation quality is not entirely clean. At sequence 54
(20:55:59 Phoenix), cumulative duplicates rise from zero to five and counter-gap
occurrences from zero to one. At sequence 60 (20:56:05), they reach ten and two.
The associated primary observer receipt intervals reach 3055.118 and 3041.483 us,
followed by receipts as short as 0.280 and 0.271 us. This is consistent with an
observer pause and queued-notification catch-up, but does not prove a matching
pause in acquWFS or instrument frame loss. `counterGaps` counts nonunit changes,
not the number of missing frames; zero `dropped` rules out collector queue
overflow, not every form of missed observation. Gap/duplicate qualification
reduces publication interval counts to 3993 and source interval counts to 3992
in affected windows; no qualified pair bridges those discontinuities. The
omitted intervals remain unmeasured, including any simultaneous trigger gaps.

For the same seven separated windows, primary interval directed fractions are
36/53 late-then-early and 14/68 early-then-late; camera publication fractions
are 37/57 and 36/89. These do not imply a universal ordering or exact
compensation, and the populations are not matched between camera and acquWFS.

Only this record and the main plan were updated for the analysis, preserving
the preceding capture and startup correction. Whitespace checks passed; no
functional changes, test/build rerun, commit, push or agent instrument action
were performed. No early-arrival reserve or scheduler timing policy is chosen.

## 2026-10-23 OCAM source review moved to its own plan

At the operator's request, the hard review of `ocam2KCtrl`, `edtCamera` and
`frameGrabber` is recorded in
[ocam2k-acquisition-publication-timing.md](ocam2k-acquisition-publication-timing.md).
The operator confirmed current `bin2` operation, 64 DMA buffers, requested
priority 90/cpuset `camwfs`, depth-one output and the two-second/20000-entry
timing settings. The new plan distinguishes those configuration facts from
unverified actual OS scheduling and installed SDK details. It records the exact
acquisition/publication order, current SDK timestamp/version caveats, timing
buffer ownership/FPS-reset concerns, failed-result and SDK-zero-ID issues,
other focused correctness findings, and proposed bounded stage measurements
with tests and operator comparisons. No camera or shared C++ implementation
was changed, and no agent instrument action, build/test rerun, commit or push
was performed for this planning task.

Subsequent operator confirmation: the installed EDT PDV version is **5.6.2.0**.
The OCAM plan now records that as confirmed and targets matching 5.6.2.0
headers/source/documentation for the timestamp, acquisition and threading
contract review. Separately loaded driver/build details and version-specific
API semantics remain to inspect; the installed PDV version itself is no longer
an open question. No implementation or instrument action accompanies this
documentation update.

## 2026-10-23 OCAM timing implementation

The operator approved the OCAM plan for implementation. The workstation change
now repairs framegrabber timing ownership, adds optional camera-stage and serial
observations, and implements the focused acquisition, SDK-lifecycle,
configuration-request and scheduling-reporting fixes. Detailed design, test
results and outstanding qualification are maintained in the
[OCAM plan](ocam2k-acquisition-publication-timing.md); the interface is described
in the [framegrabber timing guide](../../../libMagAOX/app/dev/doc/frameGrabberTiming.md).

Existing camera `atime`, realtime `writetime` and notification ordering are
preserved. Stage collection defaults off, and the write-clock bracket criterion
remains undecided by default. The always-active timeout-result guard introduces
an SDK query whose cost needs a pre-change-binary baseline comparison in addition
to stage-off/stage-on trials. This work does not choose a synco margin, change
DMA buffering/polling or software-binning pixels, or perform instrument actions.

The operator then made the current local config available and began copying
`EDTpdv` to the workstation. Bin2 is confirmed as serial mode 3 with no extra
software binning, raw 1056x62/8-bit/eight-tap input, `fv_once: 1` and
`method_framesync: EMULATE_TIMEOUT`. The copied install log records PDV 5.6.8.0
on November 12, 2025; the operator clarified that a manual install overwrote the
default. The earlier 5.6.2.0 report is retained as historical evidence, with the
manual SDK tree now the contract/build target. Loaded driver details still
require operator evidence.

The prioritized SDK files are now available. Source inspection confirms the
default seconds/nanoseconds timestamp format, latest-buffer selection and
timeout/frame-sync behavior. The camera app compiles and links against the
copied 5.6.8.0 headers/library. All 16 focused suites pass (5,737 assertions in
109 cases), as do ASan/UBSan and TSan stress checks; the full verification record
is in the OCAM plan. The matching SDK also documents same-thread DMA resource
ownership, so the existing cross-thread lifecycle remains a separately scoped
follow-up. Runtime provenance, instrumentation overhead and write-bracket
qualification remain operator work. No deployment was performed. The operator
approved committing and pushing the verified changes.

## 2026-10-23 scheduling error-reporting prerequisite

The OCAM review found that `MagAOXApp::threadStart()` interpreted positive
`pthread_setschedparam()` error codes as success. The isolated prerequisite
checks `rv != 0` and logs that returned code, preserving inherited-policy
continuation. Its offline regression passes 56 assertions in one case,
including EPERM/EINVAL and success with independent errno values. This does
not change scheduling policy or establish the instrument's actual policy.
The operator approved committing and pushing this fix with the camera work.

## 2026-10-23 first OCAM stage telemetry

After rebuilding, the operator supplied three valid `[fgstage]` windows with
stage collection enabled and no reported acquisition/frame/diagnostic losses.
SDK acquisition-interval deviation is 0.319–0.346 us, versus publication's
1.088–2.018 us; output extrema reach 424.882–572.732 us. SDK return intervals,
conversion/load and main-notification durations contain candidate tails, but
their independent extrema do not locate a particular output excursion.

`stageBracketUs=0` leaves every interval ambiguous, so zero late/early counts
are expected rather than evidence of no excursions. A provisional diagnostic
bracket of 1 us is proposed for the next unchanged-workload operator capture,
given observed brackets of 0.040–0.100 us. The scheduler reserve remains
undecided. Completed serial operations overlap about half the observed frames;
this is not proof of blocking or causation. Full populations, provenance and
limitations are in the [OCAM plan](ocam2k-acquisition-publication-timing.md).
No code, configuration, commit, push or agent instrument action accompanies
this documentation analysis.

## 2026-10-23 qualified OCAM excursion attribution

The next operator capture sets the provisional clock bracket to 1 us. All
3999 intervals qualify in each of three nonoverlapping two-second windows,
with four late and six early intervals in total and no new failures, frame
gaps or diagnostic losses. The matched cohort identity holds in all six
cohorts. Conversion/load changes contribute approximately +36.8 us to both
single-event late cohorts and -36–38 us to the early cohort means.

The largest late interval decomposes as **577.591 = 540.140 + 37.451 us**:
40.140 us above nominal is already present at SDK return, and conversion/load
accounts for 36.780 us of the additional return-to-write change. Its preceding
post-write tail is also long, at 43.743 us. Thus the evidence identifies more
than one elapsed-time region; it does not establish extra algorithmic work
inside the descrambler rather than interruption while it runs. Ordering of
classified late/early events and serial correlation remain unavailable in
this excerpt. Detailed populations and limitations are in the
[OCAM plan](ocam2k-acquisition-publication-timing.md).

Only engineering documentation changed. No code/configuration, test/build
rerun, commit, push or agent instrument action accompanies this analysis.

## 2026-10-23 longer OCAM stage capture

The operator's `camwfs_fgstage.txt` has 35 disjoint two-second windows with
140000 retained frame observations and 139965 qualified intervals: 69 late,
84 early and zero ambiguous. Record endpoints span 69.889 seconds; approximately
50–57 ms gaps between retained windows limit continuous event coverage. All
failure/frame-gap/invalid/drop counters remain zero. Two new wide brackets are
accounted cumulatively outside the retained windows.

Pooled SDK acquisition, return and publication deviations are respectively
0.359, 0.869 and 1.604 us; acquisition extrema still reach 480–525 us. The
longer sample changes the emphasis from conversion to a recurring disturbance
in multiple regions: rearm, validation, conversion, main notification and sync
notification all have tails with roughly 36–39 us added elapsed time. One
single late event has a 536.773 us output interval with +36.749 us rearm change
while conversion decreases 1.321 us. Another is already late at SDK return.
Conversion remains the largest pooled post-return contributor, but is not the
unique delay location.

Interruption/preemption or shared-resource delay is a working hypothesis,
not an identified cause. Ordered excursion/neighbor records and a controlled
scheduler/IRQ comparison are the next useful causal measurements. Serial
association and universal late-then-early ordering remain unresolved. The
[OCAM plan](ocam2k-acquisition-publication-timing.md) records numerical pooling,
matched examples, input provenance and qualifications. No C++ changes,
configuration changes, commit, push or agent instrument actions accompany
this analysis.

## 2026-10-23 causal measurement setup implemented

At the operator's request, the camera collector now optionally emits bounded
`[fgtrace]` excursion records with two preceding and two following frames,
using the existing producer event timestamps. Selection, median maintenance and
serialization run on the ordinary consumer. The causal preceding-interval median,
raw bracket qualification, exact monotonic clocks, PID/TID, identities, missing
neighbors and diagnostic losses are recorded. `stageTraceMax=16` is the initial
operator setting; zero remains the default. No additional acquisition-thread
clock reads or locks are introduced.

An operator-only helper captures a short private tracefs `mono` instance with
scheduler/IRQ activity on the camera CPUs and saves placement, event definitions
and trace-loss evidence. The
[procedure](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md) gives the rebuild,
configuration, baseline/on comparison and capture commands. The
[OCAM plan](ocam2k-acquisition-publication-timing.md) records implementation,
scope and qualification limits.

Workstation checks pass 4013 assertions in 72 C++ cases, seven fake tracefs
tests, sanitizer stress checks, normal library/reader builds and the camera
build against copied PDV 5.6.8.0. Hardware overhead and causal correlation remain
operator measurements. No configuration change or agent instrument action was
performed. On 2026-10-24 the operator approved committing and pushing the
verified follow-up: measurements/helper/tests and engineering records first,
then the user guide and capture procedure in a documentation-only commit.

## 2026-10-24 camera IRQ correlation identifies a delay source

The operator supplied `camwfs-trace-20261024T215606Z`. Its retained kernel
suffix directly correlates the recurring approximately 36 us camera stage
extensions with EDT IRQ 405 executing on acquisition CPU 33. The camera is
FIFO 90 and experiences no runnable-task displacement in that suffix.
Of 127 EDT handlers interrupting acquisition, 122 wake only the ordinary
camera control thread. Every long conversion, rearm, validation and
main/sync-notification stage in the 120 selected neighboring frames covered
by the trace contains an EDT interrupt. Source inspection confirms that
serial reception uses this same device interrupt; serial housekeeping is
the strongly supported origin of the bursts, although the running ISR's
internal implementation is unavailable.

The 8 MiB kernel ring overwrote 103364 events, retaining the final 6.69 of
ten requested seconds. Every 16-context frame-trace batch also saturates.
Approximately 56 ms gaps between two-second summary windows systematically
omit most of the short serial disturbance bursts. These selection and
coverage limits prevent extrapolating to every excursion or using the
summary RMS as continuous full-population jitter. There are no new camera
frame/counter gaps, publication failures or source queue losses.

The retained chronological sample also disproves universal classified
late-then-early ordering: 233 of 560 qualified early centers follow normal
intervals. A scheduler must not assume every early event follows a late one.
The proposed next controlled comparison separates the EDT IRQ from the
camera worker CPU, preserving serial polling and measuring both CPUs; it
has not been performed. The [OCAM plan](ocam2k-acquisition-publication-timing.md)
records exact counts, clock coverage, source evidence, matched examples and
follow-ups. Only engineering documentation changed; no implementation,
configuration, commit, push or agent instrument action accompanies this analysis.

## 2026-10-24 EDT IRQ moved to CPU 61

The next operator capture, `camwfs-trace-20261024T222140Z`, confirms IRQ 405
on CPU 61 while acquisition remains FIFO 90 on CPU 33, with the same camera
process, session and stream. The operator confirmed that only the IRQ moved;
camera settings, other CPU placement, loop settings and workload were unchanged.
In retained summaries with kernel tracing absent,
publication-interval deviation decreases **1.864 to 0.995 us**, approximately
47%; SDK acquisition deviation remains 0.431 us. Mean acquisition-to-write
latency increases slightly, **52.744 to 53.600 us**. Large conversion and
notification extensions disappear from the recorded neighborhoods.

The remaining long segment is SDK rearm. Among 98 unique neighboring frames
with joint kernel coverage, **all 26 long rearm calls overlap a remote EDT
handler**. The call finishes about 5.46 us after that handler ends, close to
the ordinary 5.30 us rearm duration. Extra rearm time averages 33.50 us versus
33.34 us of remote-handler overlap, with no local IRQ or thread switch during
the delay. This strongly suggests driver/device serialization across CPUs;
the current driver's precise locking mechanism remains unverified. The SDK
path leads from `pdv_start_image()` to the `EDTS_STARTBUF` ioctl.

CPU 61 retains its full ten seconds, but CPU 33 overwrites 50936 events and
retains 7.97 seconds. Diagnostic context drops fall to about 1%, with no new
camera frame/counter gaps, publication failures or source queue losses.
The approximately 56 ms summary-window gaps persist. The next focused
measurement is acquisition ioctl entry/exit alongside both CPUs' IRQ events,
followed by selective driver investigation if the excess is inside the call.
The [OCAM plan](ocam2k-acquisition-publication-timing.md) records exact coverage,
comparisons, a matched timing example and qualifications. Only engineering
documentation changed; no implementation, configuration, commit, push or
agent instrument action accompanies this analysis.

## 2026-10-24 OCAM rearm moved after both publications

At the operator's request, eligible multi-buffer OCAM acquisition now queues
its replacement request at the end of the existing `frameGrabberPostPublish`
hook, after both main-image and sync notifications. Timeout and frame checks
remain before publication. Other EDT controllers and ineligible single-buffer,
single-frame or serial-triggered modes retain immediate rearm. Pending requests
are handled once or discarded on recovery/exit.

Independent `rearmEnter` and timeout-query endpoints separate SDK rearm from
`timeoutQuery`, validation and notification durations. The updated readers
retain the older stage/trace formats. The timeout query is still a driver
ioctl before publication; the next operator comparison must determine whether
the remaining contention appears there. The [OCAM plan](ocam2k-acquisition-publication-timing.md)
records the implementation and verification. Hardware qualification remains
operator work; no agent instrument commands are performed.

Final workstation checks pass **4642 assertions in 86 cases** across collector,
OCAM, generic framegrabber integration, telemetry compatibility, CRED2 and
lifecycle suites. Library/schema, SDK-linked camera and `logdump` builds pass;
formatting, whitespace and documentation-link checks pass. Rebuild the camera
and reader for the next operator capture; no new configuration setting is
needed. The operator approved committing and pushing the verified changes on
`jrmales/dm-syncopation`, with implementation and engineering records first,
followed by user documentation.

## 2026-10-24 first operator comparison after deferred OCAM rearm

The local copied capture `camwfs-trace-20261025T034126Z` is October 25 UTC,
the evening of October 24 in Phoenix. Camera acquisition remains FIFO 90 on
CPU 33, with EDT IRQ 405 on CPU 61. Apart from the camera/reader rebuild and
restart, the operator confirmed no other settings or workload changes,
including during the first file's changing timing spread.

The intended ordering is observed. Six kernel-covered rearms take
**23.113–24.306 us** while overlapping remote EDT control handlers, but begin
after the current write and both notifications. Timeout queries in the twenty
covered neighboring frames take **0.230–0.481 us**, including five queries
that remain short while a remote control handler is active. Contention has not
simply moved into the prepublication guard in these samples.

Mean SDK-acquisition-to-write time improves **53.600 to 50.167 us** in retained
kernel-off summaries; selected excursion contexts decrease **416 to 19 off**
and **403 to 9 on**, with no new diagnostic selection drops or camera
frame/counter gaps. This does not establish uniform jitter improvement:
first-file pooled publication deviation is **2.224 us**, versus **0.995 us**
previously, during an unexplained rise and fall in conversion/publication
spread. Most second-file windows are **0.757–0.879 us**, but a large burst
makes that whole-file deviation **1.864 us**.

Uncovered publication intervals of **614.931 us** in the first file and
**799.700/924.626 us** in the second are already delayed at SDK return.
The first pair falls in a summary-window gap. The second burst is about
100 ms before tracing starts, within the helper's metadata/setup envelope;
that association does not identify its cause. CPU 33 still overwrites
50350 kernel events, retaining 7.98 s of the requested ten seconds, although
all four selected contexts inside the requested interval survive.

The [OCAM timing record](ocam2k-acquisition-publication-timing.md) contains
input checksums, qualification counts, pooled comparisons, matched rearm/IRQ
examples and the uncovered events. Deferred rearm achieves the current-frame
publication objective; the residual SDK-return excursions and varying first-file
spread remain open. A future comparison can separate trace setup from capture
and shorten the kernel window to avoid overwrite. No scheduler margin is
selected. Only engineering documentation changes; no implementation,
configuration, commit, push or agent instrument action accompanies this analysis.

## 2026-10-24 settled five-second capture setup

At the operator's request, the camera kernel helper now completes metadata and
private trace setup, waits **five seconds with tracing off**, then captures
**five seconds**. Both waits are configurable, with finite bounded arguments.
Explicit phase anchors bracket metadata, setup, settling and trace enable/disable;
interrupted phases remain identifiable. Saving and metadata queries stay outside
the measured interval. The shorter duration targets the observed ring overwrite,
while settling separates setup from measurement without assuming that setup
caused the earlier uncovered burst.

The [operator procedure](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md)
retains a full telemetry baseline and overlapping telemetry covering setup,
settling, capture and the following tail. Use the current camera TID and both
the acquisition/EDT IRQ CPUs, currently **33,61**. Updating this helper requires
no camera/reader rebuild or camera restart. The
[OCAM timing record](ocam2k-acquisition-publication-timing.md) records the
interface and local verification. Hardware comparison remains operator work;
no agent instrument action, commit or push accompanies this change.

Local verification passes **17 offline fake-tracefs tests**, Python syntax/compile,
CLI help, whitespace and **46 relative documentation links**. Tests qualify
normal phase ordering, interrupted preparation/capture, failed start/stop and
absence of diagnostic I/O during the timed interval. Independent review passes.

### 2026-10-24 commit/push checkpoint

The operator approved committing and pushing the helper/tests and engineering
records on `jrmales/dm-syncopation`, followed by the procedure in a separate
documentation commit. Verification remains **17 passing offline tests** and
the checks above; no functional changes followed verification. The next
instrument capture remains operator work and needs no camera/reader rebuild
or restart for this helper-only update.

## 2026-10-24 setup burst isolated by the new phase anchors

The operator supplied `camwfs-trace-20261025T040828Z` and confirmed no other
settings or workload changes. Camera PID/TID, inode/session, FIFO 90 on CPU 33
and EDT IRQ 405 on CPU 61 are unchanged. The helper completed five-second
settling but still requested **ten seconds of capture**, overwriting **50203**
CPU 33 events. Joint retained coverage is **7.991974 s**. The operator agreed
to repeat with `--settle-seconds 5 --seconds 5`.

The large **1138.810/359.769 us** publication pair occurs inside the explicit
private trace setup phase, before settling or trace enable. The SDK acquisition
and return intervals already contain the excursion; query, conversion and
deferred rearm remain ordinary. This narrows the previous metadata/setup-envelope
association to setup, without identifying its individual responsible operation.
The early interval only partly compensates: the two SDK acquisition intervals
sum **1500 us** rather than nominal **1000 us**. Consecutive counters and later
normal cadence do not establish unchanged frame age or prove absence of backlog.

The kernel-off file has **0.782 us** pooled publication deviation and mean
acquisition-to-write **49.289 us**. The entire on file has **2.084 us** deviation;
excluding its one explicitly identified setup-containing summary gives
**0.804 us**. Four summaries wholly inside the requested enabled interval also
give **0.804 us**, and settling summaries give **0.757 us**. The old first-file
broad spread does not repeat, but this baseline improvement precedes the helper
and cannot be credited to settling. No new quality failures, frame/counter gaps
or diagnostic drops occur.

The two selected contexts during tracing form a **524.841/479.184 us** pair,
already **525.152/479.355 us** at SDK return, with short query and publication
stages. Their six neighboring frames have full retained kernel coverage. The
pair lies in a summary-window gap, so requested-inside aggregate summaries
classify zero excursions. Full CPU 61 acquisition-waking spacing stays within
**480–525 us** during tracing. These are host delivery observations, not physical
exposure or absolute frame-age measurements.

The [OCAM timing record](ocam2k-acquisition-publication-timing.md) contains
checksums, exact phase/coverage boundaries, population qualifications and
matched IRQ evidence. Only engineering documentation changes; no implementation,
configuration, commit, push or agent instrument action accompanies this analysis.

## 2026-10-24 five-second repeat retains both CPUs without loss

`camwfs-trace-20261025T041847Z` confirms the intended **five-second settling /
five-second capture** sequence. Camera PID/TID, stream/session and CPU/IRQ
placement match the preceding run. Both CPUs report **zero overwrite, commit
overrun and dropped events**; joint retained event coverage is **5.000001 s**.

The earlier millisecond setup burst does not recur. Whole-file publication
deviation is **0.791 us off / 0.800 us on**, and mean acquisition-to-write
**49.256/49.393 us**. Only one complete two-second summary lies inside the
five-second trace; its deviation is **0.764 us**, not a full-five-second RMS.
The on-file raw summary population retains two ambiguous intervals from one
**1.102 us** write bracket; there are no other new quality failures, camera
frame/counter gaps or diagnostic drops.

Two selected capture contexts are **2.057 seconds apart**, with publication
intervals **524.781/479.806 us**, already **524.811/480.016 us** at SDK return.
They are not an adjacent pair. The early center follows a **509.823 us normal**
publication interval, again ruling out universal classified late-then-early
ordering. Both contexts fall in summary-window gaps and have ten distinct
neighboring frames with complete kernel coverage.

Three covered rearms take **23.264/24.396/23.525 us**, each overlapping a remote
EDT control handler. They begin after both notifications and finish on average
**5.288 us after IRQ exit**, with no local interrupt or thread switch. Covered
timeout queries remain **0.230–0.370 us**, including short queries concurrent
with remote control IRQs. This confirms continued rearm serialization after
publication. Full-trace acquisition-wake intervals span **481–524 us**, with
waking-to-run maximum **4 us** and no runnable-camera displacement.

The [OCAM timing record](ocam2k-acquisition-publication-timing.md) records
input checksums, exact boundaries, qualification counts, full/subset comparisons
and matched rearm/IRQ examples. The capture procedure now retains the intended
history; the earlier setup transients and physical frame age remain open.
Only engineering documentation changes; no implementation, configuration,
commit, push or agent instrument action accompanies this analysis.

## 2026-10-24 proposed finer EDT request attribution

Existing matched IRQ/SDK endpoints already split the two covered examples:
the late event's IRQ entries are 525 us apart, while the early event has 500 us
entry spacing and a handler-duration change of -20 us. The next proposed
drill-down prioritizes selective driver completion/timestamp/serial observations,
after confirming loaded build identity and available tracing support.
Individual EDT request timings can resolve SDK/driver gaps in other cases.
The copied SDK's wait-return bracket
includes buffer/timeout queries, blocking WAITBUF, wait-status handling and
timestamp retrieval. All actions share one outer Linux ioctl command, so the
capture must also identify the nested EDT action. This can separate pre-wait
contention, acquisition waiting, post-wake driver requests and user-space SDK
work, accounting for the existing scheduler/IRQ history.

Verified driver probes or the vendor's existing trace ring could provide the
following split inside the driver, but runtime support, clock semantics and
overhead need verification. Camera/card versus delayed host interrupt delivery
still needs an independent hardware reference if the delay precedes IRQ entry.
The [OCAM record](ocam2k-acquisition-publication-timing.md) records exact request
actions, source findings and the proposed sequence. This is not implemented;
there are no new instrument actions or configuration changes.

## 2026-10-24 complete EDT driver copy inspected

The complete extraction is at `/home/jrmales/Source/opt/EDTpdv`. It contains
Linux wrapper source, a symbol-bearing proprietary core object and `edt.ko`
version **5.6.8.0**, built for the captured **5.14.0-362.8.1.el9_3.x86_64**
kernel. The linked module retains symbols/debug information, giving concrete
completion, serial, wakeup and shared-lock probe targets despite the core C
source remaining absent.

Static inspection places ordinary `atime` creation in `edt_increment_done`
through `EdtTraceTimeStamp`, using CLOCK_REALTIME quantized to **1 us**. When
the PDV device/serial callback executes, it precedes this completion stamp in
the normal IRQ path. Variable driver work can therefore move `atime` without proving variable
physical camera delivery. Rearm and IRQ critical sections use the same channel
zero lock path; this supplies a concrete target for measuring the observed
serialization, rather than proving its runtime wait/hold durations.

The [OCAM record](ocam2k-acquisition-publication-timing.md) records module
identity/hashes, static call paths and qualified tracing targets. Loaded-module
identity is verified in the next checkpoint; probe eligibility remains an
operator check. No vendor binary,
module or probe was run; only engineering records change.

## 2026-10-24 loaded EDT module match confirmed

The operator's exao2 readbacks match the complete copied module: kernel
**5.14.0-362.8.1.el9_3.x86_64**, EDT **5.6.8.0**, source version
`A26E0B4FD875E93168B83BC`, and build ID
`97857cff4fc9d75717d3b58e9c5bc3359a7204f0`. The build ID is the final
20 bytes of the supplied sysfs GNU note, after its header/name.

The [OCAM timing record](ocam2k-acquisition-publication-timing.md) records the
comparison and next read-only capability checks. This resolves the module
identity prerequisite for targeted driver tracing; actual symbol exposure,
probe registration support and measurement overhead remain unverified.
No implementation or instrument configuration changes accompany this checkpoint.

## 2026-10-24 EDT probe prerequisites available

The operator confirms all four requested kernel configuration flags are `y`,
the root-owned `kprobe_events` interface exists, and all ten selected EDT
functions appear in `/proc/kallsyms`. Kernel lockdown reports **none**.

The next operator step is the new
[`check-edt-probes.py`](../../../apps/ocam2KCtrl/scripts/check-edt-probes.py)
registration-only helper: test entry/return definitions with events disabled,
then remove its own definitions. No active driver timing capture is enabled
by this change. The [OCAM record](ocam2k-acquisition-publication-timing.md)
records exact prerequisites and the separation between registration eligibility
and later active measurement qualification.

The helper's **16 offline checks pass**, including failed/short writes,
interruption, output failure and selective cleanup. Independent review also
passes; actual operator registration and active measurement results remain
pending. The existing scheduler/IRQ capture helper is unchanged.

## 2026-10-24 kernel policy explains partial probe registration

The operator check finishes with **20 attempted, 12 errors**. Both entry and
return definitions register disabled for `edt_wake_up`,
`edt_wake_up_interruptible`, `edt_start_intr_critical` and
`edt_end_intr_critical`; all eight are successfully removed. The six other
functions reject both definitions with `EINVAL`; no cleanup error is reported.

Follow-up configuration confirms `CONFIG_DYNAMIC_FTRACE=y`,
`CONFIG_KPROBES_ON_FTRACE=y`, and unset `CONFIG_KPROBE_EVENTS_ON_NOTRACE`.
The kernel log names every rejected function with the explicit reason
`Could not probe notrace function`. This resolves the rejection cause: the
unmodified proprietary object lacks the ftrace instrumentation present in
the accepted Linux wrappers. It is a tracefs event-registration restriction,
not a missing symbol or helper command-format failure.

The [OCAM timing record](ocam2k-acquisition-publication-timing.md) records the
accepted/rejected table, kernel source reference and revised first capture:
use the accepted lock/wakeup wrappers with native IRQ/scheduler events, qualify
probe overhead and pairing, and preserve uncertainty about uncovered lock
owners and the physical completion boundary. The ordinary clock wrapper
`edt_do_gettimeofday` is another possible observation point but remains
untested. Active probe collection is not yet implemented; this checkpoint
updates documentation only.

## 2026-10-24 accepted EDT wrapper capture ready for operator test

`capture-camera-timing.py --edt-wrappers` now adds the four accepted lock/wakeup
functions, with pointer-valued entry fields and `r64` return probes. The helper
reuses registration ownership, enables only its private instance, records
formats/profile snapshots/activation anchors and preserves unsaved buffers with
their disabled decoder definitions. Native capture keeps its existing behavior.

Wrapper mode uses 32 MiB per selected CPU and limits capture to 1–5 seconds.
The [operator guide](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md) starts
with a one-second native comparison and one-second wrapper capture, each with
five seconds of settling and overlapping camera telemetry. Hooks are active
during settling and on unrecorded CPUs; profile counts cover activation through
deactivation. Return-pool miss semantics remain unverified, so complete pairing
and loss/overhead checks remain required.

All **53 offline tests pass** (31 capture, 22 owner), including interruption,
partial setup, unrelated-state preservation and failed save/cleanup. Independent
review passes. The [OCAM record](ocam2k-acquisition-publication-timing.md) records
the exact fields, ownership choices and limitations. Copy both updated Python
scripts for the operator test; no camera/reader/kernel rebuild is required.

## 2026-10-25 first wrapper capture confirms the shared EDT lock

The operator supplied `camwfs-step5-20261025T155627Z`: one-second native and
wrapper traces after five-second settles, followed by helper-free camera
telemetry. Both retain CPUs 33/61 without loss or cleanup errors. The operator
confirms an overnight camera power-off followed by normal power-cycle FSM
startup, with no software reconfiguration. The camera cools toward −45 C
during the first two files and is near its setpoint in the last.

**14,439 complete wrapper call pairs** and **6,147 lock cycles** identify one
lock pointer shared by acquisition and IRQ 405. All **25** camera-side
acquisition brackets above 2 us overlap a remote hold; the maximum is **34 us**.
Seven IRQ-side brackets above 2 us likewise overlap the camera's hold, up to
5 us. A selected **40.957 us rearm** includes that 34 us wait and remains
after publication. These are instrumented elapsed brackets on a 1 us trace
grid, not exact uninstrumented spin times.

The ordinary main/control wake returns into `pdv_DeviceIntPdv`; completion
IRQs containing it average **53 us**, versus **32.183 us** for acquisition-only
IRQs. For captured raw frames 1349872/1349873, `atime` intervals are
**526 / 479 us**. The late interval has 526 us between shared IRQ entries;
the early interval has **500 us** between entries while handler duration falls
**53 to 32 us**. This accounts for the early software-stamp interval without
establishing an early physical frame. It does not explain every late interval.

Native tracing sees the same longer combined-handler class, approximately
50.714 versus 30.494 us. Wrapper timing is measurably higher, but file-wide
comparison is confounded by pre-existing conversion variation and the cooling
startup. Both raw excursion centers fall between summary windows. The final
helper-free retained summaries have **0.829 us publication deviation**;
that value is not a continuous bound or a summary of the wrapper ring.

A separate untraced **613 / 387 us** SDK/publication pair has steady
**500 / 501 us `atime`**, leaving a roughly 113 us post-stamp delay unassigned.
The next proposed experiment is to change routine `temp`/`gain` status
polling cadence with the camera settled and compare control IRQs, lock waits
and timing. With synchronization on in these captures, `getFPS()` uses the
synchronization rate and does not query `fps` over serial. The responsible
individual commands and a suitable polling policy
are not yet established. No runtime or scheduler change accompanies this
analysis. The [OCAM record](ocam2k-acquisition-publication-timing.md) contains
input hashes, exact timelines, phase/coverage limits and reproducible analysis
artifact locations.

## 2026-10-25 proposed bounded pause of housekeeping serial traffic

The operator proposes a testing mode to pause housekeeping and raises
phase-aware housekeeping scheduling. The recommended first test is a finite
runtime pause of routine `temp`/`gain` polls (`fps` as well when synchronization
is off), with normal / paused / normal comparisons at settled temperature.
The pause begins after an outstanding transaction drains, exposes actual
boundaries and stale readback ages, automatically expires and clears on
power/reconfiguration/error transitions. Explicit serial commands end the
quiet interval; acquisition, FSM and telemetry continue.

Gain polling also detects the software-visible `HV` protection-trip response,
so its suppression must be explicitly reported. Request timing alone cannot
guarantee deconfliction: response bytes and device/control interrupts arrive
later, potentially across many camera periods. Keep serial waits off the
acquisition worker. The [OCAM plan](ocam2k-acquisition-publication-timing.md)
records the proposed mode and later per-query scheduling measurements. This
is planning only; no testing mode or housekeeping scheduler is implemented yet.

## 2026-10-25 120-second housekeeping diagnostic implemented

The operator approved the pause with a 120-second timeout for 60-second
telemetry sets. `camwfs.housekeeping_pause.toggle=On` suppresses automatic
`temp`/`gain` and unsynchronized `fps` serial queries after any in-flight command
finishes. Off resumes, repeated On does not extend the current deadline, and
the ordinary loop automatically expires it. Acquisition, frame timing,
telemetry and nonserial FPS updates continue. Explicit serial commands and
camera lifecycle/reconfiguration changes end the quiet interval.

Read-only `housekeeping` and `housekeeping_status` expose the deadline, actual
monotonic boundaries, serial/suppressed-poll counters, readback ages/validity
and software protection-polling status. Cached readings remain stale until
successfully refreshed. Cooling-Off validation refreshes pre-pause temperature
before applying its existing guard. No generic EDT code, frame hot path or
production polling schedule changes.

The [operator guide](../../../apps/ocam2KCtrl/doc/camera-timing-trace.md)
section **7** provides the normal / paused / normal capture. This requires an
`ocam2KCtrl` rebuild/restart, with no configuration or `logdump` change. The
[OCAM plan](ocam2k-acquisition-publication-timing.md) records lock ownership,
transition semantics and local verification; instrument qualification remains
an operator follow-up.

Local checks pass: **1,026 assertions in 49 test cases** across the new
housekeeping suite and existing OCAM, lifecycle and temperature suites. The
normal app builds/links against the copied EDT 5.6.8.0 SDK without test stubs;
that binary is not launched. Formatting, whitespace and local documentation
links pass. The operator comparison has not yet been run.

## 2026-10-25 closed-loop housekeeping pause comparison

The operator supplied `/home/jrmales/tmp/camwfs-housekeeping` and confirms the
loop was closed with unchanged gain, camera settings, CPU/IRQ placement and
workload throughout normal / paused / normal captures. All three retain
session 5, inode 3383, bin2, sync ON, 2 kHz, EM gain 595 and shutter OPEN.

Serial attempts remain **981** through the paused capture; timing records show
zero control traffic/overlap. Automatic expiry resumes polling approximately
**120.426 s** after activation. Every paused two-second summary lies safely
inside the quiet interval, away from either edge. Temperature/gain are correctly
marked stale; the paused temperature is not a fresh thermal measurement.

Retained `atime` ranges are **481–525 / 499–501 / 480–515 us**. Normal files
emit **12 / 5** qualified raw >20 us contexts, all overlapping serial work;
none are emitted while paused. The prior shared EDT handler trace places
serial/device work before DMA completion stamping and wakeup. Together these
support delayed software reporting of frame readiness; physical IRQ assertion
and exposure timing are not measured.

Overall publication deviation is **0.761 / 1.428 / 0.747 us**, using 29 / 30 /
29 disjoint two-second windows. The 60 paused summaries overlap at about one
second spacing, so they are not all pooled. The pause removes large stamp
tails but a separate late conversion change raises publication RMS: first
selected paused windows have **0.656 us** publication deviation, final selected
windows **2.700 us**, with conversion **27.474 +/- 1.385 us** while SDK-return
interval deviation stays **0.270 us**. Rearm maxima fall from **27.151 / 31.019 us**
in normal files to **9.809 us** while paused. No new frame failures, gaps or
queue drops appear. Sixty-one wide clock brackets during the paused change
affect attribution qualification but do not filter the aggregate distributions.

The [OCAM record](ocam2k-acquisition-publication-timing.md) contains hashes,
actual pause boundaries, raw/context coverage limits and per-stage detail.
The next diagnostic can target post-SDK variation with a short matched kernel
trace during paused operation. Per-query attribution and a production
housekeeping schedule remain future work. This checkpoint changes only the
two engineering records; all analysis runs locally.


### Production OCAM housekeeping policy — 2026-10-25

Implemented the approved [separate housekeeping policy](ocam2k-housekeeping.md)
in `ocam2KCtrl`: no steady gain/FPS queries after confirmation, configurable
10-second idle temperature reads and faster change/convergence monitoring,
with one automatically scheduled serial query per loop. The policy record and
operator guide describe bounded confirmation, conservative cooling-state
inference, fresh cooling-Off validation and the retained diagnostic pause.
Local checks pass (2,037 assertions / 67 cases and the EDT SDK build); the
operator's hardware timing comparison remains pending. This implements the
production schedule identified above, without changing the evidence or the
remaining post-SDK conversion investigation.


### OCAM housekeeping control verification — 2026-10-25

Following the FPS/gain reply-parser corrections, the operator verified that EM
gain changes, internal FPS changes with synchronization off, and temperature
changes work on hardware. Temperature queries appear near **1 Hz while
changing** and **0.1 Hz on target**, matching the intended default schedule.
The [housekeeping record](ocam2k-housekeeping.md) records this initial hardware
qualification and the remaining checks: gain/FPS query silence, synchronized
FPS, failure/lifecycle/cooling-Off cases, and matched timing comparisons.


### Full-loop computational baseline — 2026-10-25

The operator supplied `/home/jrmales/tmp/full-loop-20261026T023744Z` and confirms
closed AO loop at gain zero, dark/unaligned optics and no changes during the
two-minute capture. The [separate analysis](full-loop-timing-20261025.md) records
coverage, hashes, statistical method and a plot. Primary publication has
**5.273 us** interval deviation; combined publication **7.099 us**; DM SDK return
**7.137 us**. DM receipt-to-SDK-return is **108.840 +/- 0.531 us**, with no new
controller errors or gaps.

Of 53 late primary intervals in disjoint qualified populations, 48 are followed
immediately by early intervals. The main associated duration changes occur in
WFS preparation and its handoff to reconstruction. These are the next AO-path
investigation targets. Camera stamp tails remain associated with sparse serial
exchanges; an independent conversion variation persists between exchanges.
Synco's camera observer has coverage gaps while the direct camera collector is
clean, so its receipt stalls must not be labeled camera frame loss. Scheduling
remains disabled and the early-arrival reserve remains undecided.

The operator subsequently confirms dmcomb was restored to **trigger mode 4**
for regular operations before this capture. Its sleep-and-counter-scan behavior
is consistent with the variable primary-to-combiner start interval
(**6.399 +/- 3.469 us**) and larger combined-publication deviation; the capture
does not isolate polling as the sole cause. Earlier mode-3 results are separate
conditions. This clarification does not change attribution of the upstream
primary excursions; the current requested polling delay remains unverified.


### Full-loop mode-3 repeats — 2026-10-25

The operator supplied `full-loop-20261026T033656Z` and
`full-loop-20261026T035040Z` under `/home/jrmales/tmp` and confirms **only dmcomb
trigger mode changed** from mode 4. Both retain closed AO loop at gain zero,
the same camera settings, CPU/IRQ placement and GPU workload, with no changes
during either capture. The [separate comparison](full-loop-trigger-modes-20261025.md)
records hashes, quality, sample selections and a plot.

Primary publication deviation is **2.168 / 2.496 us**, combined publication
**2.180 / 2.509 us**, and DM SDK-return interval deviation **2.359 / 2.531 us**,
versus **5.273 / 7.099 / 7.137 us** respectively in the mode-4 baseline.
The largest absolute SDK-return interval departures from 500 us are
**26.725 / 28.340 us**, compared with **37.646 us** in mode 4. Linked
primary-to-combined delay is **9.233 +/- 0.164 / 9.236 +/- 0.164 us**, versus
**11.589 +/- 3.404 us**: delay variation is about twentyfold lower.

Upstream preparation and source cadence also improve; the earlier broad camera
conversion regime is absent in both repeats. Sparse serial-associated camera
tails remain. Direct camera/controller collectors are clean; the separate
combiner observer has one coverage gap in each capture. Controller component
durations still change regimes despite better output cadence. Scheduling stays
disabled, so these observations qualify the measured baseline, not active
extra-step delivery or an early-arrival bound. The auxiliary-trigger route and
reserve remain separate work.


### Mode-4 return confirms regular jitter and captures a long wait — 2026-10-25

The operator supplied `full-loop-20261026T040946Z` and
`full-loop-20261026T041221Z` and confirms closed loop at gain zero with **only
the trigger mode changed** back to 4. The [comparison and plot](full-loop-trigger-modes-20261025.md)
now cover the full mode-4 / mode-3 / mode-4 sequence. Linked primary-trace to
combined-publication deviation reverses from **3.404 -> 0.164 -> 3.392/3.404 us**.
DM SDK-return interval deviation returns to **6.398 / 7.051 us**, versus
**2.359 / 2.531 us** in mode 3. The reversal supports polling as the source of
the regular additional combiner-path timing spread.

The second return contains a separate large camera event: at about
**04:13:56.043 UTC**, driver acquisition intervals remain **500 / 500 us**, but
camera SDK-return intervals are **825.058 / 174.550 us**. The current camera SDK
wait lasts **784.822 us** after normal wait entry; previous rearm/tail and
current conversion remain ordinary. Driver stamp to SDK return rises from
about 22 to 347 us during serial overlap. A trace is needed to distinguish
driver/library waiting and OS scheduling in this bracket.

The publication excursion propagates through the AO path, with additional
processing variation, reaching combined intervals **197.242–890.382 us** and
DM SDK-return intervals **196.491–890.813 us**. Controller service remains
ordinary, and direct camera/DM counters show no frame/counter gaps or failures.
This observed **+390.813 us** departure supersedes any interpretation of the
earlier 28 us maximum as a general bound. The regular mode-4 spread persists
when the outlier's whole selected window is excluded. A separate camera
conversion regime also returns late in this file. All auxiliary scheduling
remains disabled; no implementation or reserve change is made here.


### Resume fixed-grid scheduling and independent skips — 2026-10-25

The operator elects to stop short of major additional EDT work and return to
dmSyncopation, describing the pending policy as an average step-0 time with
frame skipping. The [main plan](dm-syncopation.md) now spells out the proposed
counter-labelled normal-phase grid, retaining the previously selected median
period, and independent passive/active eligibility. Published targets stay
fixed; actual main arrivals establish output associations and reader holds.
Excursions cancel or skip affected auxiliary work without moving later targets
or automatically requiring another successful passive preparation.

The plan corrects an important interpretation: the historical approximately
80 us short intervals violated the old previous-actual-arrival predictor but
do not directly measure early phase relative to a fixed grid. Add separate
phase-residual evidence before qualifying the still-undecided physical
early-main reserve. Proposed phase-estimator settings and immutable provider
plans are recorded for review. The next increment starts with shadow grid/skip
diagnostics and deterministic tests, followed by runtime integration and the
previously proposed targeted mode-3 auxiliary wake. No C++ or live configuration
is changed at this planning checkpoint.


### Shadow simulation implemented — 2026-10-26

The operator authorizes simulation of the normal-phase grid and independent
skips. Startup-only `simulation.enabled=true` now observes genuine primary and
combined metadata while operational enable remains Off; On requests are
rejected. The shadow callbacks do not attach the scheduler's DM resources,
copy destinations, publish command counters, create auxiliary/provider images,
sleep for scheduled slots or send trigger posts. Existing diagnostic collectors
continue independently. The [guide and capture procedure](../../../apps/dmSyncopation/doc/dmSyncopation.md#capture-a-simulation-run)
and [two-step profile](../../../apps/dmSyncopation/dmtweeter_synco_sim.conf)
describe the operator retest. Rebuild both `dmSyncopation` and `logdump`.

The fixed-storage model uses the median of 32 eligible primary publication
intervals and counter-normalized observation phase, with gain 0.1 and at most
1 us correction per newly committed cycle. Already committed plans remain
fixed. Outputs can arrive at the observer before their matching primary
callback; bounded exact-counter links retain that evidence without inventing
a primary event. Ambiguity poisons acknowledgements, and known newer main
evidence cancels obsolete work. Grid/period freshness, conservative actual and
virtual reader holds, the configured cadence cap and per-step eligibility
produce conditional would-run/skip counts. Arrivals inside modeled copy or
reader windows are reported separately as possible overlaps.

The measured 885.783/202.112 us test sequence produces fixed-grid residuals
**+385.783 / +87.895 us**: both main observations remain late even though the
second interval is short. The period stays 500 us, and the next active target
does not slide. This is a deterministic replay of those intervals, not a new
instrument result. Observation phase includes monitor/callback/lock delay;
source readiness and copy/delivery execution remain explicit model assumptions.
The physical early-main reserve stays **undecided**, with the configured 20 us
default recorded as an unqualified simulation assumption.

Workstation checks pass **4,772 assertions / 78 cases**, plus library/schema,
app and reader builds, C++ formatting and whitespace checks. The application
integration tests verify untouched command data/counters/semaphores, absent
new auxiliary/provider streams and zero timed waits. Existing live scheduler
and trigger routing remain unchanged; physical overhead and prospective active
delivery require the operator capture before further operational changes.


### Resume shadow capture after RTC cpuset migration — 2026-10-26

The operator merged the cpuset PR into `dev`, merged `dev` back into
`jrmales/dm-syncopation`, and reports the RTC migration succeeded with closed
AO-loop operation. The pending test remains the two-step read-only shadow
simulation; no live auxiliary delivery or targeted-wake test is selected yet.

Prepared the workstation copy of
`/opt/MagAOX/config/dmtweeter-synco.conf` with `[simulation] enabled=true`,
retaining `dm01disp03`, `maxSubframes=2`, the underscored timing-stream name
and `shmimMonitor.cpuset=dmtweeter-synco` (target CPU41). The operator must
sync this config, rebuild/install `dmSyncopation` and `logdump`, then restart
the instance. Simulation observes automatically with operational enable Off;
auxiliary providers remain stopped.

Keep the current dmcomb trigger mode and its resolved named primary, AO gain,
camera settings, CPU/IRQ placement and workload fixed for the first capture.
The mode-3 wake-route experiment is separate. After warmup, record initial
and final INDI snapshots plus 120 seconds of simultaneous `dmtweeter-synco`,
`dmtweeter` and `camwfs` telemetry. Record the actual trigger mode and AO gain
with the supplied result. Check estimator validity/period, actual output
associations and observation quality before interpreting modeled per-step
would-run, skip and overlap counts. The early-main reserve remains an
unqualified assumption. No new instrument measurement is recorded here.

#### INDI snapshot stalls during capture setup

The operator reports that the capture exceeds two minutes and suspects the
preceding `getINDI camwfs` query never returns, possibly a regression in recent
OCAM changes. The supplied shell recipe took unbounded INDI snapshots before
starting its timed collectors and used a bare `wait`. A stalled snapshot could
therefore prevent telemetry collection from starting; a bare `wait` could also
wait for unrelated background jobs in the operator's shell. The
[capture procedure](../../../apps/dmSyncopation/doc/dmSyncopation.md#capture-a-simulation-run)
now starts collectors first, bounds every snapshot externally and waits only
for the three saved collector PIDs. Partial snapshot output and exit statuses
are retained. This contains the capture delay; it does not resolve a possible
definition/request loop.

Workstation source review confirms that the previous fix, `2bab1b28`, remains
present: dmSyncopation does not force camera subscriptions once per second.
OCAM housekeeping ages/countdowns use ordinary Set messages. The repository's
`getINDI` ignores those messages by default; only matching readable property
definitions reset its wildcard-query timeout. OCAM's new explicit request is
for the external synchronization-frequency property once per FPS confirmation
generation, not for camera properties. The base class still forces subscribed
requests when answering a broad property query as server-restart recovery;
that behavior predates these OCAM changes. No repeated camera-definition path
has been established from this review.

Requested an operator-only, eight-second `getINDI -vvv camwfs` capture, saving
values and XML separately, plus `type -a getINDI` and the exit status. This will
distinguish repeated definitions from ordinary updates or differing installed
client options. Instrument evidence and a causal regression finding remain
pending. No runtime code, scheduling policy or camera polling was changed.

#### Bounded XML capture confirms repeated FPS and gain definitions

The operator supplied `~/tmp/camwfs-getindi-values.txt` and
`~/tmp/camwfs-getindi-xml.txt` on the workstation and confirms that `getINDI`
resolves directly to `/usr/local/bin/getINDI`, with no alias. The terminal
exit status was not included. Parsing the XML yields 79 complete property
vectors: 31 initial definitions, 16 additional definitions and 32 ordinary
updates. Only `camwfs.fps` and `camwfs.emgain` repeat, each with eight additional
definitions over approximately eight seconds. Each of `fg_timing`,
`fg_stage_timing`, `fg_timing_quality` and `housekeeping` sends eight ordinary
Set updates. Every other property has only its initial definition.

The repeated FPS/gain pair explains why the device-wide `getINDI camwfs`
query cannot reach its two-second quiet interval. FPS remains 2000 and gain
remains 1, with the same old property timestamps on every definition. Those
timestamps describe unchanged property contents, not response arrival times;
the nearby ordinary-update timestamps establish the approximate repetition
cadence. The replies do not demonstrate renewed camera serial polling.
The narrower `getINDI camwfs.fg_timing` query is expected to finish because
its definition does not repeat in this capture; that expectation has not yet
been checked on the instrument.

Audited OCAM, `stdCamera`, the INDI message serializer, base subscription
handling, the camera GUI, numeric status widgets and GUI discovery retries.
The current camera code replies to targeted GetProperties requests with these
definitions; no periodic FPS/gain definition sender was found. Ordinary Set
messages retain their type through serialization. GUI discovery and base
subscriptions stop retrying resolved properties. `strehlEstimator` subscribes
to both FPS and gain, and `modalGainOpt` subscribes to gain, but neither current
source establishes a periodic request for this pair. A requesting client,
different installed code or another request path remains to be identified.

Requested the connected GUI/script/notebook inventory and whether
`strehlEstimator` was running. The client XML does not include requester
identity. Do not restore serial polling or change the AO/camera settings to
mask the definition traffic. No runtime fix is justified yet; the bounded
capture procedure remains available for the pending shadow test.

The operator confirms `strehlEstimator` is running, together with cameraGUI
and rtimvClient camera overlays on possibly three or four control stations.
Review of current and historical client code found no matching periodic
FPS/gain query path; multiple connected clients alone do not establish the
cause. The operator proposes isolating the remote machines by shutting down
`isAOC` and `isICC`, leaving `isRTC` running. The local configurations support
this test: AOC imports RTC devices through its remote-server configuration,
and ICC imports `camwfs` explicitly from RTC.

For the first isolation capture, the operator stops those two remote INDI
servers, leaves RTC applications including `strehlEstimator` running, allows
the disconnect activity to settle, and repeats the bounded camera XML query
on RTC. If the repeated pair disappears, restore one remote server at a time
and repeat after connection/discovery settles to identify the implicated
route. If it persists, restore both remote servers and investigate RTC-local
requesters or clients connected directly to RTC. A direct workstation-to-RTC
tunnel bypasses this isolation, and reconnect bursts must not be mistaken for
the persistent repetition. No isolation result has been supplied yet.

#### Repetition persists with AOC and ICC servers stopped

The operator supplied `~/tmp/camwfs-rtc-only-values.txt` and
`~/tmp/camwfs-rtc-only-xml.txt` and confirms that the problem persisted during
the remote-server isolation. The capture contains 82 complete vectors: the
same 31 initial definitions and 16 repeated definitions, plus 35 ordinary
updates. FPS and EM gain are again each defined nine times with unchanged
payloads and the same timestamps as the first capture. All other definitions
occur once. Disconnecting the AOC and ICC INDI-server routes therefore did
not remove the repeating source; those routes are not necessary for the hang.
RTC-local subscribers and any direct RTC connections remain candidates.

The next operator test restores the remote servers and temporarily stops only
the `strehl` instance, retains the existing camera/AO/GUI state, and repeats
the bounded XML capture on RTC before restarting `strehl`. This tests whether
the local app mediates the FPS/gain requests; a positive result would still
require distinguishing an app-internal request from requests induced by
another client's queries to that app. No runtime code change is made.

The operator subsequently reports that the repetition continues with both
remote INDI servers still down and `strehl` stopped. This rules out requiring
that app or either remote-server path; no new XML capture or exit status was
provided for this condition. Further transport review found no automatic
FPS/gain query in the INDI connection timers or byte-copying FIFO bridge.
The next diagnostic is a bounded passive capture of incoming TCP port 7624
traffic with before/after `ss` connection/process snapshots, retaining the
current isolation during capture. This can identify directly connected
requesters without further app shutdowns. Requests from local apps through
driver FIFOs will not appear in that TCP capture; an SSH-forwarded connection
may require tracing its forwarding endpoint afterward. Requester identity
remains unresolved and runtime code remains unchanged.

#### Operator identifies the unmerged data-collection app

The operator identifies `dlDataCollection`, whose deployed changes are not
yet merged, as the source of the repeating FPS/gain requests. This resolves
the requester investigation for the reported INDI hang; the passive TCP
capture is no longer needed. Neither the checked-in app nor the identical
copy in `MagAOX-working` contains the deployed camera-subscription code, so
no speculative change is made to either copy. No new post-fix XML capture
has been supplied.

Correct the repeated requests in that app's development work. A normal
subscription should retain Def/Set updates and use the base class's
unresolved-property discovery/restart handling rather than repeatedly force
definitions of resolved camera properties. The bounded snapshots remain
useful protection for the capture recipe. Restore the temporarily stopped
INDI servers and `strehl` for the normal test configuration, then resume
the two-step read-only shadow capture once the request loop is stopped.
No OCAM polling, timing policy or runtime source change is required by this
finding. The definition flood and its containment are separate from the
pending dmSyncopation scheduling-policy measurement.

### First RTC shadow capture — 2026-10-26

The operator supplied `~/tmp/synco-sim-20261026T215557Z`, recording checkout
`aaeef915915b6ed8159b29e3c50c089f954af52f`. The 119 `[dmsynco]` records span
21:56:03.506–21:58:02.134 UTC (118.628 seconds). Matched files contain 120
DM-controller timing records and 114 camera stage-timing records, each
covering about 119–120 seconds. They contain live capture records, not the
older history suggested by their binary telemetry filenames.

The app reports shadow mode, operational enable Off and zero delivered
auxiliary commands. All 119 simulation records have a valid grid, nominal
feasibility and session 3, with two modeled steps and 32 period/phase samples.
Reported median periods range from 499.088 to 500.936 us. Top-level live
scheduler fields remain inactive (`schedule_valid=0`), as expected in this
mode. The configured 20 us early-main reserve remains an unqualified model
assumption.

**The capture does not yet qualify the skip/copy policy:** no actual
main/combined-output association was accepted. Every combiner report has
trigger inode zero and primary counter zero despite a readable trace from
writer PID 3850720. Across the INDI snapshots, combined events and
`unlinkedPrimary` both increase by 253,337; simulation `outputGaps` also
increases by 253,337 while `associations` stays zero. Trace metadata errors,
missing traces, collector drops and the existing combiner duplicate/gap
counters do not increase. Here, `outputGaps` describes unusable model
associations, not lost physical DM commands.

First-to-last telemetry counter differences cover 237,255 additional primary
observations and give the following modeled outcomes:

| Step | Considered / skipped | Early-main reason | Missing-output reason | Would run / overlap |
| --- | ---: | ---: | ---: | ---: |
| Passive 0 | 237,255 / 237,255 | 64 | 237,191 | 0 / 0 |
| Active 1 | 237,255 / 237,255 | 0 | 237,255 | 0 / 0 |

The reason counters follow the model's gate order. These are not qualified
physical skip rates or evidence that overlap would be absent with working
output associations. The INDI before/after interval is longer than the
first/last telemetry interval and instead brackets 253,337 new primary
observations; do not mix those populations. Existing session phase extrema
−228.309/+122.455 us are already present before the capture and remain
unchanged. The 1 Hz sampled residuals range from −23.723 to +18.240 us, but
cannot establish per-frame residual tails.

Independent timing measurements remain usable. The following RMS values are
medians of reported overlapping two-second-window standard deviations, not
whole-capture RMS; ranges are the union of those windows' extrema.

| Measurement | Median window RMS, us | Observed range, us |
| --- | ---: | ---: |
| Camera acquisition interval | 0.324 | 480–525 |
| Camera publication interval | 0.841 | 478.273–525.151 |
| Primary DM publication interval | — | 473.474–526.955 |
| Combined DM publication interval | 5.825 | 468.143–531.614 |
| DM receipt to SDK return | 0.433 | 107.853–114.135 |

The primary publication RMS ranges from 2.355 to 3.897 us across its windows;
do not subtract independently sampled RMS values to infer combiner latency.
DM cumulative events, receipts and successful submissions each increase by
238,342 between its first and last records, without new failures, partial
commands, duplicates, counter gaps, invalid events or collector drops.
Camera events/publications increase by 239,598 without new failure/gap/drop
counts. Twelve additional serial controls and six unique qualified camera
excursion traces occur; each trace's five-frame context reports serial
overlap. The trace intervals include one 525.151 us late interval and five
approximately 479 us early intervals. Existing trace-drop counts do not grow.

Saved settings show loop state On and gain 1, unlike the earlier gain-zero
comparisons. Camera records retain bin2, 2 kHz, EM gain 1, shutter SHUT,
synchronization On and CCD target −45 C throughout. The actual dmcomb trigger
mode/name/delay are not established by the capture. Requested those FPS
readbacks and confirmation of any setting/placement/workload changes during
the run. Zero trigger inode alone does not identify the trigger mode.

Next, establish a resolved named primary in dmcomb's trace. A blank,
unresolved or stale-at-initialization `.procinfo.triggersname` can produce
this symptom; verify the running values before changing anything. A resolved
`dm01disp03` (or its `aol1_dmC` alias) can supply linkage while retaining mode
4. After normal reinitialization if needed, require a matching nonzero inode,
advancing primary counter and growing `simulation.associations` before
another long policy capture. The guide now makes this preflight explicit.
No runtime source or instrument setting was changed during this analysis.

#### Correct FPS readback; live trigger identity still to be checked

The operator reports dmcomb trigger mode 4, trigger-stream name `dm01disp03`
and delay `0.00001` seconds (10 us). These values are compatible with the
required linkage; the configuration name is not blank. Confirmation of other
setting/workload changes during the capture has not yet been supplied.

The APP-side audit confirms that the zero inode/counter comes directly from
combined trace row zero. `dmCombTiming` exports those identity fields before
its expected-primary test, and the simulation independently reads the same
row. A failed expected-inode comparison does not manufacture these zeros.

Local MILK source distinguishes configured FPS values from initialized RUN
state. `fps_connect.c` reads the name on RUN connection;
`CLIcore_utils.h` resolves `image_ID(triggerstreamname)` once at initialization.
`processinfo_waitoninputstream_init()` saves the resolved metadata inode, or
stores inode zero and replaces the live name with a space if resolution
fails. Its mode-4 branch does not clear a valid inode. Runtime updates refresh
mode/delay/timeouts but do not re-resolve name/ID/inode. Output publication
copies that saved live identity into trace row zero. The inspected dmcomb
loads numbered channel streams before initialization, so a correct existing
primary name should normally resolve at startup. Stale or failed
initialization is a supported hypothesis, not an established diagnosis.

Requested the actual dmcomb processinfo row through `milk-procCTRL` F4
(Triggering), matching captured writer PID 3850720 or its current replacement.
If live identity is blank/zero, distinguish startup resolution from current
FPS values before normal reinitialization. If it is nonzero/matching while
output trace telemetry remains zero, inspect producer/observer trace handling
instead; do not treat restart as a proved remedy. A different identity calls
for checking the FPS-instance/PID association. The guide now includes this
read-only check. No mode, CPU, timing reserve or runtime source was changed.

The operator confirms live processinfo has inode zero, a blank stream name
and mode `4:DELA`, after restarting since the earlier online mode-3 tests.
The active RUN therefore has no resolved primary lineage, matching the
captured trace zeros. This establishes the missing linkage, but not why the
current FPS name differs from the initialized state. Mode 4 itself does not
clear a valid inode in the inspected implementation.

The operator elects to set the primary name manually and restart. Retain
mode 4 and the 10 us delay, set/save `dm01disp03` before RUN initialization,
then verify the live F4 name/inode and increasing simulation associations
before repeating the shadow capture with operational enable Off. Naming the
primary supplies provenance while DELAY mode retains all-channel polling.
Do not weaken the simulation association gate or infer associations from
nearby timestamps. No restart result has been supplied yet.

The operator subsequently reports that all post-restart preflight checks are
good: the live named-primary linkage is present, simulation associations
increase and operational enable remains Off. The next step is another
120-second matched shadow capture with mode 4, the 10 us delay and other
camera/loop/placement/workload settings held fixed. Its new before/after
snapshots will separate the linked run from the earlier unlinked counters.
No post-restart timing results have been supplied yet.

### RTC latency setting added to the recorded baseline — 2026-10-26

The operator specifies `sysMonRTC.set_latency.toggle=On` as required for best
RTC performance and confirms that it has been On for **all tests so far**.
This applies to the earlier camera/EDT, full-loop, trigger-mode and shadow
comparisons; it is an operator-confirmed historical condition, not a new
setting change between those runs.

The main plan and user-guide preflight now require this state and holding it
fixed. The bounded capture recipe saves the exact property before and after
the telemetry interval, including query statuses, so future captures carry
the readback explicitly. This documentation change neither operates the
instrument nor adds a runtime enforcement mechanism.

### Linked shadow capture — 2026-10-26 Phoenix

The operator supplied `~/tmp/synco-sim-20261027T003510Z.tgz`. Its workstation
extraction is nested under
`~/tmp/home/jrmales/tmp/synco-sim-20261027T003510Z`; no source files or archive
contents were changed for analysis. The recorded checkout is still
`aaeef915915b6ed8159b29e3c50c089f954af52f`. All 118 `[dmsynco]` records have a
valid, nominally feasible two-step grid, session 3, enable Off and zero actual
auxiliary deliveries. They span 2026-10-27 00:35:17.697–00:37:15.299 UTC
(2026-10-26 17:35:17.697–17:37:15.299 Phoenix), or 117.601 seconds.

**Primary/combined linkage is now working.** Every record identifies writer
PID 3948918, trigger inode 4676 and an advancing named primary counter. The
INDI before/after interval has 251,287 new primary observations and combined
events, 251,285 accepted simulation associations and two new model output
gaps. The large old unlinked/output-gap totals are inherited: `unlinkedPrimary`
does not increase. Combiner duplicate/gap, invalid/missing metadata/trace,
rejected and dropped counters do not grow. The two new lost model
associations are not evidence of missing physical DM commands.

| INDI before/after outcome | Passive step 0 | Active step 1 |
| --- | ---: | ---: |
| Considered | 251,286 | 251,287 |
| Would run | 251,018 (99.893%) | 251,278 (99.996%) |
| Skipped | 268 | 9 |
| Overlap evidence increments | 40 | 125 |

This shows high conditional availability, **not clean timing qualification**.
Overlap counts are retrospective observations inside an already modeled copy
or active reader-hold window; would-run decisions are not retroactively
converted to safe skips. A combined-first and primary-later observation can
both count the same main. These are evidence increments, not unique unsafe
frame counts or measured physical collision probabilities.

The population boundaries matter. Between the initial INDI snapshot and the
first saved telemetry record, about seven seconds pass and step 0 adds 266
skips/39 overlap increments; step 1 adds one skip/125 overlap increments.
Those events cannot be localized individually from the retained 1 Hz rows.
They must not be discarded from full-snapshot reporting or attributed to
warmup or to the snapshot queries without evidence.

The narrower first/last telemetry interval has 235,202 considered slots per
step. Step 0 would run 235,200, skips twice for missing output, and adds one
overlap increment. Step 1 would run 235,194 and skips eight times: two missing
output and six reader hold, with no new active overlap increments. The
missing-output increments first appear at 00:35:44.839 and 00:35:57.907 UTC;
the passive overlap increment first appears at 00:36:11.978 UTC. These are
aggregate-report times, not per-event timestamps. Independent combiner quality
counters do not change at the two lost model associations, so their exact
`badOutput()` cause remains unresolved.

Median estimated periods range from 498.811 to 500.846 us. The 1 Hz sampled
phase residuals range from −17.297 to +16.010 us and cannot establish the
per-frame tails. Session extrema −504.964/+214.716 us are already present in
the initial snapshot and remain unchanged. Keep the 20 us early reserve and
5 us observation allowance explicitly unqualified. The model also assumes
10 us copy/snapshot budgets, 20 us delivery, 115 us keepout, 20 us guard,
1 us visibility and 2 us preparation reserve. No physical auxiliary copy,
wake or delivery is performed by this run.

Camera and controller timing remain similar to the earlier unlinked capture.
Values below are medians of reported overlapping two-second-window standard
deviations, not pooled whole-capture RMS; extrema are the union of those
windows' extrema.

| Interval | Prior median RMS, us | Linked-run median RMS, us | Linked-run extrema, us |
| --- | ---: | ---: | ---: |
| Camera acquisition | 0.324 | 0.281 | 480–525 |
| Camera publication | 0.841 | 0.789 | 478.623–524.290 |
| DM combined publication | 5.825 | 5.802 | 468.555–529.520 |
| DM receipt | 5.879 | 5.859 | 468.315–530.051 |
| DM SDK return | 5.916 | 5.869 | 469.096–530.292 |

The 120 valid controller records span 119.170 seconds in session 4; its change
from the prior session 3 occurs between captures, not during this run. Its
cumulative events, receipts and successful submissions each increase by
238,339 without new failures, partial commands, duplicates, counter gaps,
invalid events or collector drops. Camera session 8 continues across 112
valid windows and a file rotation; events/publications increase by 234,588
with no new failures, frame/counter gaps, invalid events or collector drops.
Eleven serial controls and five unique qualified camera excursion traces are
recorded, each with serial overlap in its five-frame context. One trace has
a 524.300 us interval and four have approximately 479 us intervals. Trace
records and rolling aggregates are separate populations.

DM processing duration becomes more variable over approximately
00:36:14–00:37:12 UTC, with receipt-to-callback window RMS reaching 1.966 us
and SDK mean reaching 102.045 us. No contemporaneous command failure/gap is
reported. Aggregate timing populations alone do not establish a cause for
the modeled overlap or for this processing variation. Linked primary-to-
combined-publication means are 11.654–12.098 us, with window RMS
3.250–3.430 us and extrema 5.590–19.005 us.

Saved camera/loop settings match the prior capture: loop state On, gain 1,
bin2, 2 kHz, EM gain 1, shutter SHUT, synchronization On and CCD target −45 C.
Mode 4 / 10 us and the latency-On requirement are operator-reported setup;
the original eight-file recipe does not include a `sysMonRTC` readback. The
updated guide captures it explicitly. The operator confirms that settings,
placement and workload remained unchanged throughout the run, including the
initial snapshot interval. The same-line simulation and combiner counters have summary times
several milliseconds apart; their numerical difference does not demonstrate
frames of physical backlog.

Recommended next diagnostic: bounded event records for overlap evidence,
including observer origin, primary/output/plan counters, step, frozen target,
copy endpoints and reader-hold endpoint, plus the observation timing needed
to distinguish lag from producer phase. Identify repeated evidence for one
main without concealing the existing aggregate counters. Capture the initial
snapshot interval as well by starting collectors first. The main plan records
this proposed diagnostic increment; the guide clarifies overlap-count
semantics. Physical margins remain undecided, live scheduling remains
disabled, and no runtime code or instrument setting was changed.

### Next shadow test: timestamped overlap evidence

Implemented the proposed trace in response to the operator's request to do
the next test. The existing fixed-grid model, gates, would-run/skip outcomes
and aggregate overlap counts are unchanged. Each overlap observation now
copies a fixed record into a 256-event FIFO under the existing cycle mutex.
The ordinary application thread removes at most 64 records per existing
`telem_dmsynco` emission, releases the mutex, then encodes/submits telemetry.
There is no formatting, allocation or telemetry submission in the monitor
callbacks. This adds bounded evidence copying and shadow-only clock reads;
its hardware timing impact still needs the matched capture.

Records retain process-lifetime event ID, original model session, observer
origin, actual primary/combined counters, modeled plan counter and step,
copy versus active-reader classification, frozen plan timing, and previous
main/output observations. The evidence main's grid can be extrapolated from
the last frozen cycle if it has no explicitly committed plan. Receipt,
callback entry, lock request/acquisition and metadata-observation endpoints
use the same-host monotonic clock. Stream publication and producer trace
start/update timestamps remain raw and explicitly labeled as a separate
clock domain; the app does not infer absolute clock conversion.

Both combined-first and primary-later evidence remain visible. Within one
app process, `(session, planCounter, step, mainCounter)` correlates repeated
evidence for one modeled step and genuine main. This preserves the existing
counters while allowing the analysis to identify repeated observations.
The last-main fields describe the previously accepted primary, not the
highest main counter seen through either monitor.

Queue quality is exposed by read-only `simulation_trace` and by the optional
appended `trace_quality`/`sim_overlap` telemetry.
`detected = drained + dropped + queued` holds across resets; a full FIFO drops new records without
changing model behavior. Pending records keep their original session across
resets/configuration changes. A failed cycle try-lock leaves them queued.
`logFailed` counts drained records lost to a detected encoding/submission
failure; it does not detect logger filtering or prove disk durability.
Legacy payloads without the appended trace remain valid. An emission that
cannot acquire a fresh queue snapshot omits trace quality rather than
claiming a zero-event batch.

The next operator capture uses the rebuilt app **and logdump** with the
current simulation configuration, operational enable Off, named-primary
association advancing, dmcomb mode 4 / 10 us, and
`sysMonRTC.set_latency.toggle=On`. Keep camera/loop settings, placement and
workload fixed. The guide's bounded 120-second recipe starts all three
collectors before initial snapshots and saves latency readbacks before and
after. Confirm `trace_quality` appears even before the first overlap; retain
every event, queue-loss counter and the initial interval. Compare copy/reader
classification, repeated identities and observer delay before drawing any
conclusion about physical timing margins. The earliest-main reserve remains
undecided and no physical auxiliary command is produced by this test.

Workstation tests pass **28,829 assertions / 87 cases**: pure model
22,972/23, application integration 4,684/51 and telemetry compatibility /
validation 1,173/13. Coverage includes interval boundaries, both observer
origins with the same correlation key, distinct main/step identities,
producer-clock preservation, FIFO overflow/drain recovery and reset
persistence, unchanged scheduling under saturation, lock contention,
64-record draining, command-image/counter/semaphore immutability, exact
integer binary/text/JSON telemetry and malformed/legacy payload handling.
All changed C++ files were clang-formatted and whitespace checks pass.
Normal library/schema, application and `logdump` builds pass. The application
and reader builds were completed sequentially after their parallel runs
raced the existing shared generated-version-header removal target; no build
system change was made.

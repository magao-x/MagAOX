# dmSyncopation debugging and optimization record

This record tracks operator observations, diagnosis, changes and verification during debugging and optimization. The [main engineering plan](dm-syncopation.md) retains the settled design and review answers; the [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents current operation and interfaces.

Current status: the measured camera period is selected correctly, but enabled operation still produces repeated primary-interval faults. On the same instance with no other reported changes, disabling synco reduced camera acquisition jitter from 8.10 us to about 0.60 us; after stopping the process it was 0.586 us. Repeated camera-property requests explain why a property-wide `getINDI` query stayed open while synco ran; local removal of that periodic polling awaits operator retesting. INDI eventually responds after restart; the exact startup delay remains unmeasured. Instrument timing qualification remains pending.

Append subsequent investigations as dated entries, keeping operator evidence, local verification and outstanding instrument checks explicit. All agent work runs on the workstation; the operator performs instrument actions and supplies results.

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

This controlled comparison strongly associates the larger acquisition timing spread with enabled synco operation, while leaving the mechanism unresolved. Camera `acq_*` values use EDT-provided DMA timestamps; they do not independently establish physical exposure-clock jitter. The primary interval is measured at callback entry and includes publication timing plus changes in wake/dispatch latency. Enabled-only work includes repeated at-most-10-us sleep checkpoints, shorter combined-output polls, mapped copies/posts, and extra downstream DM combinations/commands. Timer/CPU interference, CPU/SMT placement and downstream processing are candidates to distinguish, not established causes. Retain the configured hard bounds rather than widening them to accept the observed excursions.

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

Acquisition jitter is close to the preceding disabled-process baseline (0.600831874 us), while the enabled snapshot was 8.101606773 us. These observations support investigating enabled scheduling and downstream work as the source of the larger spread. They do not identify which component causes it, and one stopped window does not establish a timing bound.

The operator additionally observed that `getINDI camwfs.fg_timing` never finished while synco was running. Source tracing confirms a concrete cause: `dmSyncopation::appLogic()` forcibly requested its subscribed properties once per second. Every camera `getProperties` response supplies a property definition, and indiserver broadcasts that definition to interested clients. Property-wide `getINDI` has a wildcard element and waits for a quiet interval; its default two-second alarm is restarted on each matching definition (`INDI/INDI/getINDI.c`, `listenINDI()`/`findDPE()`). One-second definition traffic therefore prevents normal query completion. This is recurring forced polling, rather than a new request generated by every received camera update; server subscriptions are deduplicated.

Remove the app-specific periodic request and its timestamp member. Retain the base class's initial, unresolved and restart subscription handling and ordinary incoming camera updates. The five-second valid-reply expiry remains unchanged: if all published values remain unchanged long enough to produce no update, use the existing qualified-primary-window/nominal fallback and reacquire on reference changes. Do not use property-definition traffic as an acquisition heartbeat. The camera mailbox and hard timing checks remain unchanged.

This change addresses the demonstrated query-timeout regression. Removing repeated outgoing definitions may also reduce traffic during the bridge's quiet-based startup drain, but startup delay and the enabled jitter increase still require operator verification.

Local verification passed **1660 assertions in 20 test cases**, including existing incoming-report, mailbox, strict-expiry, fallback and unchanged-bound checks, followed by a successful normal app build and clean clang-format/whitespace checks. Independent source review confirmed normal initial/unresolved/restart subscriptions remain registered and the app-specific periodic request is absent. The app fixture does not exercise live driver transport; query completion after deployment remains an operator check. No new test scaffolding, agent instrument action or commit was introduced.

## 2026-10-20 periodic-polling fix commit and push checkpoint

The user requested pushing the periodic-camera-polling fix to `origin/jrmales/dm-syncopation`. Package the scheduler change, main plan and debugging record together, followed by a documentation-only user-guide commit, per AGENTS.md.

Validation remains **1660 assertions in 20 passing test cases**, a successful normal app build and clean formatting/whitespace checks; no functional changes followed that verification. Operator confirmation of normal query completion and startup behavior remains pending. The enabled timing disturbance remains unresolved. No agent instrument action was performed.

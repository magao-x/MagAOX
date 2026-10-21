# dmSyncopation debugging and optimization record

This record tracks operator observations, diagnosis, changes and verification during debugging and optimization. The [main engineering plan](dm-syncopation.md) retains the settled design and review answers; the [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents current operation and interfaces.

Current status: primary timestamp excursions occur with two steps and with passive-only one-step operation. The one-step comparison added 147 primary faults between snapshots, while its later snapshot was armed. The new one-step fault is 476.199 us published versus 475.799 us observed, only -0.400 us of changing observation delay; the two-step fault was 562.743/562.813 us. Extra DM triggers are therefore not required for the primary faults, although indirect effects from synco and its later-step load remain unresolved. The operator reports high-gain loop stability and confirms no frame-behind condition in this run. AO main threads and CUDA helpers are restricted to CPUs 45/46, separate from synco's CPU 61, which has no listed SMT sibling. Disabled-synco ITER summaries show baseline process-start variation, occasionally 485–520 us from the filter onward, without yet reproducing the larger publication tails. Exact startup latency, the origin of primary publication variation and sustained two-step qualification remain open; earlier EDT-confounded observations are retained below.

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

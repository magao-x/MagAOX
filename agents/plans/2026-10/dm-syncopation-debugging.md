# dmSyncopation debugging and optimization record

This record tracks operator observations, diagnosis, changes and verification during debugging and optimization. The [main engineering plan](dm-syncopation.md) retains the settled design and review answers; the [user guide](../../../apps/dmSyncopation/doc/dmSyncopation.md) documents current operation and interfaces.

Current status: the first operator-run two-step trial did not sustain the requested cadence. The local fixes below pass regression tests and the normal app build; an operator retest and instrument timing qualification remain pending.

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

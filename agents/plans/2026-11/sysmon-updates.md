# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
sysMonitor needs to be updated to monitor GPU temperature and load, and we also need a way to track if IRQ cpu pinning is as expected.

## Discussion

sysMonitor does not currently track GPU temperature or load.  We need to add periodic recording of relevant information about this. 

We also have a problem where we need to run the move_irq script after reboot to get optimum performance, but currently have no way to check/track whether this was done and if the IRQs have stayed where they are supposed to.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

1. sysMonitor should periodically record GPU temperature
2. sysMonitor should periodically record GPU load parameters
3. sysMonitor should publish these new diagnostics on INDI
4. sysMonitor should telemeter these new diagonistics.  
5. a mechanism should exist to warn the operator if IRQ placement is not as expected.
6. IRQ placement checking should handle changes in IRQ number without requiring editing of config.


## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->
Reuse exsiting telemetry schemas if possible, but fine to create a new one if needed.


# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Status: implementation authorized on 2026-11-15 after the answers below. Verified host `mx11` and checkout `/home/jrmales/Source/MagAOX-working`. Feature branch `jrmales/sysmon-diagnostics` starts from updated `dev` at `92c20495`, including the merged IRQ policy changes. No instrument commands have been run.

## Task Summary

Extend `sysMonitor` with per-GPU temperature and load measurements, INDI publication, and periodic telemetry. Add a read-only IRQ placement check that resolves devices to their current IRQ numbers, compares their CPU affinities with configured expectations, and exposes actionable operator warnings. Detect actual placement, not merely whether `move_irqs` has run.

### Findings

- `apps/sysMonitor/sysMonitor.hpp` contains the app implementation. `appLogic()` collects CPU/disk/RAM/Chrony diagnostics; `updateVals()` publishes INDI; `checkRecordTimes()` and `recordTelem()` implement periodic forced telemetry through `dev::telemeter`. Reuse these interfaces and retain the app's header-only convention.
- CPU-load collection invokes `mpstat -P ALL 1 1`, so the existing main loop already includes a roughly one-second sampling wait. New polling intervals must be independent of telemetry intervals and evaluated with a monotonic clock; they are best-effort intervals subject to existing loop work.
- No GPU collector or suitable GPU telemetry schema was found. `telem_temps`, `telem_coretemps`, and `telem_coreloads` store anonymous vectors; the latter two specifically represent CPUs. `telem_drivetemps` adds identities but represents drives. Reusing those payloads for GPUs would lose meaning or validity information. New GPU telemetry is justified.
- `libMagAOX/sys/runCommand.cpp` waits before draining its pipes, reads at most one 4096-byte buffer per stream, does not report child exit status, has no timeout, and returns into application code after a failed child `execvp`. The installed mxlib command helper has similar limitations. GPU collection should not introduce another caller of this behavior without a bounded execution path.
- `rtSetup/RTC/move_irqs` and `rtSetup/ICC/move_irqs` discover IRQs from action names in `/proc/interrupts`. Their dedicated targets are listed below. CPU lists come from named cpusets, so the monitor can reuse the framework's `m_cpusetPath`/`cpusetPath()` instead of duplicating CPU numbers.

| Host | Action-name match | Expected cpuset |
| --- | --- | --- |
| RTC | `edt` | `camwfs_irq` |
| RTC | `cp2x72c` | `woofer` |
| RTC | `enx98b78501aebc` | `shmimTCP` |
| ICC | `enx503eaa0ceeff` | `camsci1` |
| ICC | `enx503eaa0cf4cd` | `camsci2` |
| ICC | `enx98b78501ae64` | `shmimTCP` |
| ICC | `pvcamPCIE_1` | `camflowfs` |
| ICC | `pvcamPCIE_0` | `camllowfs` |

- The merged IRQ changes remove dedicated NVIDIA pinning on RTC and ICC: NVIDIA IRQs remain with ordinary IRQs on system CPUs. Dedicated monitor examples therefore omit NVIDIA. The merged ICC comments and operator confirmation establish `pvcamPCIE_1 -> camflowfs` and `pvcamPCIE_0 -> camllowfs`.
- The scripts also try to move ordinary IRQs to the `system` cpuset: currently `0-23,64-87` on RTC and `0-31,64-95` on ICC. They tolerate failures for these IRQs. A blanket equality check for every numeric interrupt would therefore be too strict; general housekeeping-CPU isolation needs a separately agreed policy and exceptions.
- `apps/bmcCtrl/bmcCtrl.hpp::verifyWorkerThread()` already parses cpuset CPU lists, but its parser is app-local and accepts partial numeric conversions. A small strict shared CPU-list parser would serve the new checker and provide a future reuse point; changing BMC behavior is outside this task.
- `stateRuleEngine` already supports numeric/text/switch comparisons, warning repetition, and recovery notifications. Its current rules cannot act solely on an INDI property's Alert state. Publish an explicit warning value and provide a rule configuration rather than adding a notification system.
- Existing deployment files `/opt/MagAOX/config/sysMon{RTC,ICC,AOC}.conf` have CPU/disk settings only. They are outside this checkout. Prepare tracked configuration examples or a deployment patch; do not silently edit installed configuration.

## Key Assumptions

1. The first GPU backend targets NVIDIA physical GPUs. Monitor all discovered GPUs, keyed by UUID; retain PCI bus ID as context. Optional expected UUIDs can detect a card absent from the first sample. AMD GPUs, MIG instances, and per-process accounting are not initial targets.
2. Proposed GPU metrics are temperature in degrees C, GPU activity and memory read/write activity as fractions in `[0,1]` to match existing CPU loads, VRAM used/total in MiB, and graphics/SM/memory clocks in MHz. Memory activity and allocated VRAM are distinct diagnostics. These fields are supported by NVIDIA's selective-query interface. [NVIDIA metric definitions](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
3. Propose configurable five-second GPU and IRQ checks, with GPU polling explicitly enabled in deployment configuration. An empty IRQ rule list disables IRQ checking. Existing configurations retain their behavior until the new settings are added.
4. Use RTC and ICC dedicated IRQ rules listed above, excluding NVIDIA. Required devices missing from a check produce a warning; an explicitly optional, absent device is reported as absent rather than passing a placement check. General IRQ isolation is a scope question below.
5. All additions observe state only. They do not change IRQs, cpusets, CPU governors, GPU settings, services, or the existing `set_latency` behavior. Instrument qualification is performed by the user.

## Requirements

1. Collect the GPU metrics periodically and associate every value with a stable device identity.
2. Publish GPU values, availability, and identity on read-only INDI properties. Unsupported fields and failed queries must not appear as healthy zeroes.
3. Record GPU measurements and validity on first sample, change, and the existing telemeter maximum interval. Distinguish the last measurement time from a later forced record of cached state.
4. Re-resolve every configured IRQ group on every check, including all matching vectors, so IRQ renumbering and driver reloads require no configuration edits.
5. Compare dedicated IRQ requested affinity with the expected CPU set, and check that effective routing stays within that set. Expose missing devices, read failures, and unverified effective routing explicitly.
6. Publish per-group IRQ results and an aggregate warning value, log state transitions, and provide `stateRuleEngine` rules for operator notifications. Include IRQ placement snapshots in telemetry so later drift can be reconstructed.
7. A GPU/IRQ failure must not stop unrelated monitoring. Preserve existing diagnostic behavior and keep new collection work bounded where the operating system permits.
8. Verify parsing, discovery, lifecycle, INDI, and telemetry with offline fixtures and fake commands; follow the full changed-file documentation and formatting requirements in `AGENTS.md`.

## Questions and Points of Clarification

1. **GPU scope and cadence:** Are NVIDIA physical GPUs on RTC and ICC, the five metrics above, and five-second polling the intended initial scope? Recommendation: yes; leave power, clocks, throttling reasons, and temperature alarm thresholds for a later request unless needed now.

ANSWER: Yes. Except I think clocks are interesting.

2. **GPU backend:** Approve selective `nvidia-smi` CSV queries with a bounded shared command helper, or prefer NVML? Recommendation: the command backend fits this app and avoids a new CUDA/NVML build dependency. NVIDIA recommends NVML for maintained tools because CLI output compatibility is not guaranteed; strict parsing and fixtures mitigate but do not remove that tradeoff. [NVIDIA interface guidance](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

ANSWER: nvidia-smi is fine for now.

3. **IRQ coverage:** Start with the dedicated RTC and ICC groups above, or also check all other movable device IRQs against the `system` cpuset now? Recommendation: dedicated groups first. If general isolation is required now, agree how to identify unmovable/managed exceptions without hardcoded IRQ numbers before implementation.

ANSWER: dedicated groups first.

4. **Device presence and identity:** Which IRQ groups may legitimately be absent while their hardware is powered down? Is `pvcamPCIE_1 -> camflowfs` and `pvcamPCIE_0 -> camllowfs`, as used by ICC `move_irqs`, correct? Recommendation: required by default; explicitly mark legitimate optional groups. Do not resolve the PVCAM discrepancy by guessing.

ANSWER: required by default. Yes. We have cleand up the IRQs, just pulled dev to show this update.  This includes not pinning NVIDIA.

5. **Unavailable effective affinity:** Should a kernel without `effective_affinity_list` warn or accept requested-affinity-only checking? Recommendation: show an unverified status and warn by default; allow an explicit configuration exception if needed for a known host.

ANSWER: accept recommendation.

All review questions are resolved by the answers above and the subsequent instruction to proceed. Accepted additions include current graphics, SM, and memory clocks in MHz. Dedicated groups are required by default; missing effective affinity warns unless explicitly allowed. NVIDIA IRQs are not part of the dedicated rule examples.

## Tests

All automated tests use synthetic input and temporary files. Do not launch the real `sysMonitor` entrypoint, call unisolated startup, execute `nvidia-smi`, or read the instrument's proc/sys files from tests.

| Area | Cases and acceptance criteria |
| --- | --- |
| Bounded command execution | Fake executables: successful stdout/stderr, missing executable, nonzero exit, signal exit, timeout, output larger than pipe capacity, output-limit rejection, interrupted waits, and cleanup. Never return through app code in a failed child, wait for an unrelated child, deadlock on full pipes, or accumulate children after repeated failures. |
| GPU parsing | One/multiple GPUs, whitespace, reordered rows, duplicate UUIDs, empty output, incorrect field counts, `N/A`/unsupported fields, malformed/trailing/nonfinite/overflowing numbers, utilization outside 0–100, and inconsistent memory sizes. Check unit conversions and preserve valid fields when another field is unsupported. |
| GPU lifecycle | Disabled mode performs no query; interval scheduling uses injected time; first success, unchanged sample, query failure, disappearing/expected-missing GPU, and recovery produce the correct validity, identity, logging, and telemetry transitions. Enumeration order cannot swap devices. |
| CPU-list parser | Singletons/ranges, whitespace, duplicates, reordered equivalent lists, CPUs above 63, empty input, reversed ranges, negative/overflowing/trailing garbage, and bounded range expansion. Compare normalized sets rather than strings. |
| IRQ discovery | Numeric interrupt rows with varying CPU columns, nonnumeric local interrupts, multiple action names/vectors, no matches, overlapping rules, and matching restricted to the action-name portion. Idle IRQs must still be checked. |
| IRQ renumbering and races | A logical device moves from IRQ A to B; vectors appear/disappear; cpuset contents change; files vanish during a snapshot. Re-discovery must recover without config edits and an incomplete check must never publish OK. |
| IRQ policy | Correct requested singleton/set; requested mask too broad even if current effective routing is correct; effective CPU outside the expected set; valid effective subset; missing/empty/malformed/unreadable cpuset or affinity; missing required/optional devices; unsupported effective-affinity interface. |
| INDI and notifications | Read-only properties, stable GPU names, explicit invalid/unverified state, all IRQ vectors included, no repeated unchanged updates/log floods, and bad-to-good recovery. Load the example `stateRuleEngine` warning rule against synthetic properties and verify raise/clear behavior without a real server. |
| Telemetry | GPU and IRQ FlatBuffer round trips, identity/value association, validity, timestamps, corrupt-buffer rejection, message formatting, generated dispatch/accessors, and first/change/forced record behavior. Never stamp old measurements as a new successful poll. |
| Regression | Retain existing sysMonitor parser tests; injected collector failures must leave the other diagnostics operational. Exercise new app integration only with external commands and the latency thread isolated. |

After implementation, build `libMagAOX` and `apps/sysMonitor`; run the focused sysMonitor, command-helper, CPU-list, telemetry, and applicable stateRuleEngine suites. Representative existing commands are:

```sh
make -C libMagAOX
make -C apps/sysMonitor
make -C tests -f Makefile.one -W ../apps/sysMonitor/tests/sysMonitor_test.cpp t=../apps/sysMonitor/tests/sysMonitor_test
apps/sysMonitor/tests/sysMonitor_test
make -C libMagAOX/logger/tests do_catch2_test
```

Register new focused suites in `tests/tests.list` and run them through `tests/Makefile.one`; add FlatBuffers linkage only where needed. Implementation validation results are recorded in the execution notes below.

## Implementation Plan

### 1. Confirm policy and prepare a feature branch

Resolve the questions above and update this document. Before implementation/commits, use a username-prefixed branch such as `jrmales/sysmon-diagnostics`. Preserve unrelated untracked files and the user-authored portion of this document.

### 2. Add small reusable collection utilities

- Add an explicitly bounded overload/helper next to `sys::runCommand`, leaving existing callers' API and behavior unchanged. Use argv-based execution without a shell, drain both streams while waiting for the specific child, cap captured output, report exit/signal/timeout distinctly, close descriptors, and terminate/reap the owned child. Retain process ownership if termination cannot complete promptly, prevent another GPU query until cleanup completes, and keep the app loop/shutdown from waiting indefinitely. Use a spawn approach safe for this multithreaded app; do not run allocation-heavy child-side C++ after fork. The GPU command uses fixed query arguments and a known executable path, without elevated privileges.
- Propose a small `libMagAOX/sys` CPU-list utility for strict normalization and set comparisons, based on the existing BMC parsing approach. Apply it only to the new checker in this task; do not refactor BMC or thread scheduling.
- Put non-trivial utility implementations in `.cpp` files and add focused offline tests. These are supporting library changes, not a new `dev::` monitoring base class.

### 3. Add GPU collection and state

- Add enable, polling interval, query deadline, and optional expected-UUID configuration. Proposed defaults are disabled, five seconds, and a two-second command deadline; validate positive finite intervals/deadlines. Enable in the reviewed deployment examples.
- Use one query per due interval for all physical GPUs:

  ```text
  nvidia-smi --query-gpu=uuid,pci.bus_id,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.current.graphics,clocks.current.sm,clocks.current.memory --format=csv,noheader,nounits
  ```

- Parse into a temporary snapshot before updating live state. Key devices by UUID, preserve PCI context, normalize activity percentages to fractions, and keep per-field validity. A malformed whole response/failed command invalidates the current poll; a documented unsupported field invalidates just that field.
- Keep known/expected missing devices visible as unavailable. Retry failed discovery on the configured cadence. Explicitly distinguish disabled, unavailable, partially supported, and valid states; repeated identical failures log only on transition, with recovery logged once.
- Publish one read-only numeric property per UUID-derived INDI-safe key, with temperature, GPU load, memory load, used/total memory, and validity elements. Companion text properties carry UUID, PCI ID, and status. Never reuse a vanished GPU's key for a different GPU. Register new discoveries through framework helpers and publish state changes through existing update helpers.
- Preserve last good numbers only alongside invalid status after a failed read. Store the last actual sample time separately so periodic telemetry cannot imply a new measurement. This is explicit local-poll failure handling, not receipt-age expiry of subscribed INDI values.

### 4. Add IRQ policy checking

- Configure named logical groups with an action-name match, expected cpuset name (or explicit CPU list), and a required/optional flag. Use literal substring matching of parsed action names, matching the scripts' current intent; do not match interrupt counters or store numeric IRQs in config. Reject conflicting overlapping policies instead of silently choosing one.
- At each due interval, read `/proc/interrupts`, resolve all matching numeric IRQs, read expected cpuset CPU lists through the existing framework path, and read requested/effective affinity for every match. Read files directly; no shell scripts or affinity writes.
- For dedicated groups, require requested affinity set equality and a nonempty effective affinity set contained in the expected set. Linux distinguishes permitted affinity from effective delivery, so effective equality is not required for a multi-CPU expected set. [Permitted-affinity interface](https://docs.kernel.org/core-api/irq/irq-affinity.html), [effective-affinity examples](https://docs.kernel.org/core-api/irq/managed_irq.html)
- Missing/malformed/unreadable data and mid-check device removal yield an unavailable result and retry next interval. Missing required devices and placement mismatches raise warnings. Optional missing devices remain visibly absent. Missing effective-affinity support remains unverified unless a reviewed host exception permits requested-only checking.
- Publish `irq_status` text by logical group, `irq_details` with IRQ numbers and expected/requested/effective CPU lists, and a read-only `irq_warning` switch with an aggregate warning element. The aggregate is On for any required group that is unchecked, missing, unavailable, or unverified under its configured policy, and for any present group's mismatch/read failure. Optional absence alone does not warn. Only a complete successful check clears a required group's failure.
- Provide `stateRuleEngine` examples comparing the explicit warning switch to `On` with warning priority. Reuse its subscription/retry/notification behavior, including clearing; do not add periodic get-property requests. Log detailed failure/recovery transitions in sysMonitor as well.

### 5. Add identified telemetry records

- Add `telem_gpu` and `telem_irq` schemas and wrapper headers. Use `telem_drivetemps` for identified collections and `telem_pdu`/its tests for validity patterns, while defining payloads specific to these diagnostics.
- GPU records contain UUID/PCI identity, metrics, per-field validity, collection status, and last sample time. IRQ records contain logical group, match status, expected CPUs, current IRQ numbers, requested/effective CPU lists, warning/reason, and check time. Use deterministic ordering so enumeration changes do not generate spurious changes.
- Add first/change/forced record methods and include the new types in the existing telemeter scheduling. Retain invalidity and failure transitions in the historical record. Keep per-instance previous-sample state for the new diagnostics; exclude acquisition timestamps from value-change comparison so unchanged polls can be suppressed while forced records carry the latest actual sample time.
- Register event codes in `libMagAOX/logger/logCodes.dat` and dependencies in `libMagAOX/Makefile`. Allocate unused codes during implementation. Generate dispatch/accessor/schema outputs using the normal build; do not hand-edit generated files or change existing telemetry formats.

### 6. Supply configuration, documentation, and validation

- Add documented RTC/ICC sample settings and warning rules under `apps/sysMonitor/config/`, plus app usage documentation describing units, validity, intervals, read-only behavior, and deployment steps. Finalize the PVCAM examples only after confirmation. Prepare external deployment changes for review if their owning repository is identified; do not install them from this checkout.
- Complete the full-file documentation pass on touched C++ files: top `\file`/`\brief`, remove `\author`, add function summaries and inline parameter docs, document member state, and separate data/accessor named sections. Keep definitions outside the class and preserve header-based app implementation. Prefer `TELEMETER_*` interface macros when updating those hooks, checking existing error behavior before replacement.
- Put `sysMonitor_unit_test` under `application_unit_test` in `tests/groups.dox`, retain `libXWCTest::sysMonitorTest`, document every scenario, and preserve API links using `SYSMONITOR_TEST_DOXYGEN_REF` for harness-indirected methods.
- Run clang-format on touched C++ files and restore `{ //mutex scope` annotations. The existing sysMonitor header needs substantial mechanical documentation/formatting work; separate functional commits, documentation-only commits, and a final formatting-only commit as required by `AGENTS.md`. Commit the updated plan with its associated implementation decisions.
- Run the offline checks above, inspect the diff, and record results here. Do not run the instrument app as a smoke test.

Expected affected paths after approval:

- `apps/sysMonitor/sysMonitor.hpp`, `apps/sysMonitor/tests/sysMonitor_test.cpp`, new app documentation and configuration/rule examples.
- `libMagAOX/sys/runCommand.hpp`, `runCommand.cpp`, its tests, and new strict CPU-list utility/test files.
- `libMagAOX/logger/types/telem_gpu.hpp`, `telem_irq.hpp`, corresponding `schemas/*.fbs`, focused logger tests, and `logger/logCodes.dat`.
- `libMagAOX/Makefile`, `tests/tests.list`, `tests/Makefile.one` if new suites need linkage, and `tests/groups.dox`.
- This plan; a tracked deployment patch if needed after host/policy confirmation. No IRQ-setting scripts need modification for the proposed dedicated-group checker.

## Follow-up and Edge Cases

- **Instrument qualification:** The user should compare GPU readbacks with published/telemetered samples, verify warnings before/after their normal `move_irqs` procedure and reboot, and check recovery after their normal device reload/power workflow. The agent will not execute those steps. Measure loop impact with monitoring enabled/disabled, including the writer's own timing such as `fg_timing.delta_aw`, not only end-to-end latency.
- **Polling impact:** Five seconds is a proposed starting point, not a measured optimum. GPU driver queries can add contention even though they do not access loop shared-memory streams. A kernel call stuck in uninterruptible sleep cannot be made promptly reapable by a userspace timeout; the runner must avoid repeated launches while an owned child remains outstanding and report this limitation explicitly.
- **Driver/identity changes:** CSV output is not a stable cross-version API. Driver upgrades need fixture/qualification checks. MIG and unsupported metrics stay explicitly invalid. UUID identity prevents enumeration swaps; action-name changes, NIC renames, or PVCAM driver-instance renaming can still require a policy update even though numeric IRQ changes do not.
- **Affinity limits:** Requested/effective CPU masks describe allowed/current routing, not proof of which CPU handled every interrupt. Cumulative interrupt counters include earlier placement and are not used as a health test. Managed IRQs, CPU hotplug, and broad housekeeping isolation require the agreed policy above.
- **Startup/configuration:** Cpuset creation or device probing may lag sysMonitor startup. Keep an explicit unavailable warning until data is available, then recover automatically. No-rule/disabled monitoring must be distinguishable from successful checking. Invalid configured rules must not silently disappear.
- **Operational integration:** The setup repository already supplies a `move_irqs.service`; why manual rerunning is necessary is a separate diagnosis. This task reports current placement and drift rather than attempting service-order or irqbalance remediation.
- **Existing code:** Unrelated legacy parser, telemetry, and `set_latency` issues are outside scope. The shared bounded runner is a new execution path; migrating old callers and reusing the strict CPU-list parser in BMC can follow separately. The one necessary legacy correction is owned-child waiting, described below.

## Execution Notes

- Working checkout and merged base verified before edits; user-authored task and answer text preserved.
- Implemented separate `sys::boundedCommand` and strict CPU-list utilities without changing legacy `runCommand` call sites or BMC scheduling. The bounded runner retains unreaped children and prevents relaunch; destruction has a bounded cleanup/reaper fallback.
- Added UUID-keyed GPU samples with eight measurements and per-field validity, IRQ rediscovery with requested/effective checks, read-only INDI status/warnings, and separate telemetry records carrying actual collection times.
- Added RTC/ICC configuration fragments and `stateRuleEngine` warning examples matching the merged IRQ policy. No installed configuration or hardware was modified.
- The normal app compile reached linking, where pre-existing coverage-instrumented INDI archives required coverage runtime linkage. Validation uses `LDFLAGS=--coverage` with production optimization for final builds; initial integration builds also used `COVERAGE=1`. No instrument app is launched.
- Review found legacy `runCommand` used `wait(NULL)` twice; a pending GPU child could therefore block an unrelated CPU/disk query. The scoped supporting fix waits for the legacy command's own PID and removes the second wait. Its API, output behavior, and callers remain unchanged. Two offline child-ownership regressions pass (20 assertions).
- Logger integration also required static `lastRecord` definitions, runtime verification cases, and support for existing `Vector_String`, `Vector_UInt`, and `Vector_Double` metadata enum values. Focused tests verify actual FITS values, custom formatting, precision, and unavailable inventories.
- Initial validated results: GPU parser 101 assertions/4 cases; CPU-list parser 42/2; IRQ collector 122/5; GPU telemetry/FITS 90/4; IRQ telemetry 29/2; bounded runner 78/8; existing metadata 22/4; generated logger suite 469/86; Python generator 6 tests. SysMonitor integration passed 223 assertions/9 cases including strict-policy-boolean regressions; existing sysMonitor tests passed 75/6. Final documentation/formatting checks are in progress.
- Review corrected optional-only startup reporting, IRQ cadence after slow GPU queries, and policy booleans that mxlib would otherwise silently convert from malformed text to false. New diagnostics explicitly accept only `true`, `false`, `1`, or `0` for those booleans. Per-device GPU status changes are logged even when another device keeps the aggregate status partial.

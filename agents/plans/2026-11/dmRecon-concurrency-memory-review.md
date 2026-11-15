# dmRecon concurrency and memory-safety investigation

Review date: 2026-11-05. Recorded for commit: 2026-11-06. Author: Codex (GPT-6).

Request: Analyze dmRecon after on-sky operation appeared to "blow up" after several minutes, concentrating first on concurrency and memory safety in its substitution of the current integrator state in cacao mfilt. Document the findings in a planning document.

Status: Source audit and initial dmRecon-only implementation complete on 2026-11-15; workstation validation recorded below. The cause of the observed on-sky event is **not yet established**. Historical source references below remain tied to the reviewed baseline.

## Implementation decision — 2026-11-15

The user's final choice is **dmRecon-only passive observation**: monitor the existing loop-trigger stream's semaphores, publish feedback, then verify from source timestamps that the exact next trigger occurred later. Warn when publication misses that timestamp or its ordering cannot be established. The user rejected a dmRecon-controlled trigger path. Do not gate mfilt, change its trigger configuration, or delay its execution. mfilt source remains unchanged.

This supersedes both the earlier paired dmRecon/mfilt proposal and the subsequently considered producer-side exclusion requirement. Source changes in cacao, ImageStreamIO, mxlib, and common MagAOX base classes remain outside this implementation's scope; app-local tests and their required test registration are included. There is no pending gate approval or requirement to obtain an exclusion guarantee before completing this diagnostic implementation.

The observer cannot exclude a concurrent mfilt read. Reconstruction completion, a posted semaphore, FPS estimates, and a final deadline/counter check do not prevent the writer from being descheduled. A post-publication warning cannot undo a mixed vector already consumed. **F1 remains unresolved:** successful timing classification is evidence about observed publication ordering, not proof that every mfilt read is protected or that the on-sky failure is fixed.

The dmRecon work also addresses validated private snapshots, exclusive calibration/buffer ownership, coalesced input notifications and unique frame identities, serialized enable/disable publication behavior, corrected mask indices, nonfinite rejection, dimensions/type checks, inversion-error handling, and CPU/non-CUDA build coverage. Tests exercise actual app logic with synthetic/private data and never start instrument applications.

Respect the current real-time stream rules in `AGENTS.md`: preserve frame-grabber publication ordering, perform computation in private storage, reuse the existing staged/non-temporal copy options, and treat metadata reads, copy methods, and thread placement as measurable costs. Instrument A/B timing remains user-run follow-up; workstation tests establish correctness, not RTC latency.

### Implementation record and validation status

Implementation is based on MagAOX `76fbd36ebaa1790ce746f38b41d1527ac5fdfaee` after merging `dev` into `jrmales/dmrecon-concurrency-review`. The historical audit below describes the earlier baseline; its line references apply to that baseline, not the updated header.

The working implementation has these responsibilities:

- `apps/dmRecon/dmRecon.hpp`: monitors capture private float snapshots with type, shape, capacity, counter and writer-state checks. A single frame-grabber worker owns reconstruction, calibration-derived buffers and GPU allocations. Calibration/input generations and a latest-command mailbox replace cross-thread mutation of calculation buffers. Duplicate input identities are ignored; wakeups are hints rather than queued output vectors. Publication is serialized with disable and calibration invalidation. The existing feedback and diagnostic stream names remain in use.
- `apps/dmRecon/dmReconMath.hpp`: validates masks and response dimensions, selects active pixels in storage order, builds a private inverse, initializes diagnostics, rejects nonzero inversion status and nonfinite values, and retains the previous inverse object on candidate failure. Feedback remains inhibited when the current calibration generation is unusable. This does not introduce a new science-dependent condition-number or amplitude threshold.
- `apps/dmRecon/dmReconTiming.hpp`: retains a bounded history of trigger identities/timestamps, matches an exact successor counter, and classifies completed publications as `before_next`, `late`, or `unverified`. It neither controls mfilt nor reconstructs missing trigger timestamps from callback arrival time. The source-command timestamp must lie between the candidate baseline trigger and its exact successor; a stale observer baseline is unverified.
- `apps/dmRecon/tests/`: three isolated regression suites replace the unrelated template. The app harness invokes real snapshot, calibration, reconstruction, mailbox, enable/disable and publication callbacks using process-local buffers. It never starts the application, opens streams, posts an instrument semaphore, or invokes hardware APIs. Tests are registered in `tests/tests.list` and grouped in `tests/groups.dox`; the app Makefile lists both helper headers as dependencies.

The existing `MagAOX::latestSnapshot` was considered; it requires fixed-size trivially copyable values, whereas these frames own variable-sized vectors and calibration/input generations. The app therefore uses vector swaps under its existing publication/snapshot ownership instead of adding a new common handoff utility.

Runtime review corrected the clock wiring and output attachment checks. Output attachment identity comes from `fstat` on the mapped descriptor, cross-checked against the pathname; the app thread requests a reconnect on a later pathname change and inhibits further publication once detected. The check runs at app-loop cadence, not on every feedback frame. Partial worker-startup failure now joins started workers before semaphore/member destruction. The original float modal-vector schema is preserved: an existing `aolN_modevalDMf` must contain exactly the reconstructed mode count, with one column and one plane when represented as a three-axis stream. An oversized output is not compatible merely because it has sufficient byte capacity; publishing a prefix would leave stale tail coefficients.

### Workstation validation — 2026-11-15

Host verified as `mx11`, not an instrument computer. The app binaries were compiled and linked into `/tmp` and **never executed**. Only the isolated test executables were run. No instrument configuration or common dependency source was changed.

| Check | Result |
| --- | --- |
| App compile/link, installed CUDA support, `-O3 -ffast-math -Wall -Wextra` | Passed with no compiler diagnostics. GPU execution not tested. |
| App compile/link with `-UMXLIB_CUDA -UHAVE_CUDA` after installed flags | Passed with no compiler diagnostics; exercises compilation of the separate CPU branch. |
| `dmRecon_test` production build | 372 assertions / 11 cases passed, including 2,000 concurrently submitted commands, exact output schemas, malformed private payloads, finite CPU reconstruction, trigger qualification, duplicate/coalesced inputs and disable/generation boundaries. |
| `dmReconMath_test` production build | 158 assertions / 6 cases passed, including asymmetric masks, invalid response dimensions, nonfinite inputs, singular inverse and injected positive/negative SVD failures. |
| `dmReconTiming_test` production build | 135 assertions / 11 cases passed, including delayed callbacks, exact/equal deadlines, source bracketing, gaps, identity resets and timestamp anomalies. |
| Math and timing helpers, ASan + UBSan | Passed all respective assertions; LeakSanitizer disabled because this environment uses ptrace. |
| App harness, ASan | Passed with `detect_leaks=0:new_delete_type_mismatch=0`. Unsuppressed ASan found an existing logger teardown mismatch: a variable-sized `flatlogs::logHeader::createLog` allocation is freed through scalar `shared_ptr<char>`. No logger change is included here. |
| App harness, TSan | 372 assertions / 11 cases passed, no reports or suppressions, `halt_on_error=1`. GCC warns that TSan does not support `atomic_thread_fence`; this checks exercised in-process ownership, not interprocess stream coherence. |
| App harness, UBSan | Build blocked by existing `libMagAOX/logger/types/telem_fgstage.hpp:485`: instrumented member-function-pointer comparison in `if constexpr` is rejected by GCC. No full-app UBSan result is claimed. |
| Formatting / whitespace | Ran clang-format on all changed/new C++; restored mandated `{ //mutex scope` spelling. `git diff --check` passed. |

There are **665 passing assertions in 28 production test cases**. Sanitizers instrumented the test/app header translation units; existing `tests/testMain.o`, `libMagAOX.a` and installed libraries were linked without rebuilding them under sanitizers. App-startup rollback and actual stream replacement were reviewed and compiled, but the harness deliberately does not launch the application or run shared-memory lifecycle integration. These results do not establish on-sky causality or RTC latency.

The registered tests can be built individually with `make -C tests -f Makefile.one t=../apps/dmRecon/tests/dmRecon_test` (and similarly for `dmReconMath_test` and `dmReconTiming_test`). The standalone app-test build actually used here, from the repository root, was:

```sh
g++ $(pkg-config --cflags mxlib) \
  -DMAGAOX_NOEDT -DMAGAOX_GITHEAD_NOWARNING \
  -DMAGAOX_GITHEAD_NO_UNTRACKED_WARNING -Iflatlogs/include \
  -Wall -Wextra -O3 -ffast-math \
  apps/dmRecon/tests/dmRecon_test.cpp tests/testMain.o libMagAOX/libMagAOX.a \
  -o /tmp/dmRecon_test $(pkg-config --libs mxlib) -ludev -lxrif \
  libs/libtelnet/libtelnet.a INDI/libcommon/libcommon.a INDI/liblilxml/liblilxml.a
/tmp/dmRecon_test '[dmRecon][app]'
```

For app TSan, replace `-O3 -ffast-math` with `-O1 -g -fno-fast-math -fsanitize=thread -fno-omit-frame-pointer` and run with `TSAN_OPTIONS=halt_on_error=1`. For app ASan use `-fsanitize=address` instead and the explicit options above. Helper tests use the same test main and installed mxlib flags/libraries (the timing-only helper needs only C++20 and pthread); they pass with `-fsanitize=address,undefined` and `ASAN_OPTIONS=detect_leaks=0`.

Remaining validation and limitations:

- F1 is monitored, not fixed. Producer metadata checks detect observed changes but are not a portable mutual-exclusion protocol with mfilt or with other stream writers. The upstream delta's relationship to hardware completion and static-offset generations remains outside this app's proof.
- Clock-domain configuration, the exact mfilt trigger alias, and latency under the deployed cpusets need user verification. Reconstruction now runs on the **frame-grabber worker**, so its CPU affinity/priority must be evaluated alongside the command-monitor and new loop-trigger observer. The extra snapshot and observer have measurable cache/latency costs.
- Test GPU execution and failure handling on an isolated workstation setup; perform realistic calibration/control replay and sustained latency testing. The app replaces CUDA allocation objects as a whole to avoid the inspected mxlib resize/free defects; mxlib itself is unchanged. A failed GPU initialization falls back to CPU, while a failed allocation/calibration inhibits that generation.
- Output mapping replacement is detected at app-loop cadence. Optional diagnostic mappings reconnect during frame-grabber configuration; independent diagnostic-only replacement and missing-at-startup streams may require another reconfiguration. The `_diff` stream is an approximate asynchronous comparison copied into private storage, not proof of a matched mfilt/DM iteration.
- Separate mode and mask streams have no shared transactional calibration identifier. Generation ownership prevents use-after-reallocation and rejects superseded frames; it cannot prove that independently published mode/mask data belong to one external calibration transaction.
- Full process-level stream recreation, startup fault injection, GPU sanitizer coverage, realistic wind-up reproduction, and any further mfilt hardening remain follow-up work. On-instrument checks must be run by the user.

### Passive timing interpretation and configuration

The observer snapshots trigger `cnt0`, stream identity and source `writetime`. A command carries its captured baseline into the worker. Completion is sampled after the frame-grabber's main publication/semaphore post, independently of the output stream's earlier `writetime`. The exact next trigger may already be observed when completion is recorded, or its callback may arrive later; the diagnostic uses source timestamps in either case.

The required classifications are:

| Classification | Meaning |
| --- | --- |
| `before_next` | A qualified command/publication pair completed strictly before the observed exact successor's source timestamp. |
| `late` | A qualified pair completed at or after that source timestamp; equality counts as a miss. |
| `unverified` | The necessary successor, source association, identity or clock evidence is absent or inconsistent. A later counter across a gap cannot substitute for the missing successor. |
| `rejected` | An input snapshot or reconstruction frame was rejected; inverse-build failures are logged separately. This is separate from timing classification. |
| `coalesced` | A newer input superseded a still-pending command in the mailbox. |

Repeated semaphore wakes with the same counter are not additional trigger events. Latest shared-memory metadata does not preserve a timestamp for every queued semaphore token. Counter gaps, counter resets, reconnects/inode changes, contradictory timestamps, bounded-history eviction and clock discontinuities must not produce an affirmative classification. An unresolved publication is retired as unverified when superseded or explicitly reset; absence of a successor is not evidence of success. Application-thread warnings report accumulated late/unverified results without per-frame logging in the publication path.

In the inspected ImageStreamIO publisher, `writetime` is sampled before `cnt0` increments, `write` clears and semaphores are posted. Completing before that timestamp is conservative evidence of completing before its post. A `late` classification is a missed **source-timestamp deadline**, not proof of the exact semaphore-post order or of an overlapping mfilt read. Neither timestamp bracketing nor a DM delta counter proves which mfilt iteration caused that delta, particularly when auxiliary DM commands are present.

Configuration belongs in the existing dmRecon instance configuration. The intended defaults follow the local cacao files: loop 1 monitors `aol1_mfiltINPUT`, normally a symlink to `aol1_modevalWFS`; other loops default to `aolN_modevalWFS`. `loopTrigger.shmimName` must match the actual configured mfilt trigger, including any changed alias target. This implementation does not modify that alias or mfilt configuration.

```ini
[recon]
triggerClock=isio

[loopTrigger]
# Tweeter default; use the actual existing mfilt trigger stream.
shmimName=aol1_mfiltINPUT
spin=false
```

`recon.triggerClock` accepts `isio`, `tai` where available, or `realtime`. `isio` means the clock selected by dmRecon's installed ImageStreamIO headers; the local installation selects TAI, while the ordinary MagAOX frame-grabber output timestamp is realtime. Do not compare those raw timestamps directly. The command-delta source is assumed to publish with `CLOCK_ISIO`. If the selected trigger clock differs, source-command association is **unverified** unless an explicit validated conversion is implemented; choosing a clock name is not automatic discovery of the producer's clock.

Clock qualification must cover the baseline/next-trigger observations as well as command capture and publication completion. Bounded samples against a monotonic clock can detect observed offset changes; they cannot prove that an unobserved change-and-restoration never happened. An unknown producer clock or an unqualified timestamp must stay unverified. The diagnostic does not infer stability from silence, and the subscribed FPS value remains valid until updated or explicitly invalidated under the existing INDI framework.

Leave `loopTrigger.spin=false` for the requested semaphore observation. If polling is deliberately enabled later, follow `AGENTS.md`'s cpuset/SCHED_FIFO restrictions and measure its effect. Existing `framegrabber.staged` and staged-copy options remain available; the working app defaults to private staging. Added observers, copy choices and placement require user-run A/B measurements of writer timing, including `fg_timing.delta_aw`, as well as end-to-end loop timing. No instrument configuration or hardware action has been performed for this work.

## Reported operating conditions

User clarification, 2026-11-05: "I don't know the revisions. No changes in loop. The symptom is what appears to be integrator wind-up, with the DM saturating all actuators."

Further user clarification, 2026-11-05: the affected DM was the **tweeter**. The saturation observation was "all red on the display," which does not distinguish high from low limits. The rail pattern is therefore unknown; do not infer a common offset or mixed spatial pattern from the display color.

Treat apparent wind-up and apparent all-actuator saturation after several minutes as the reported symptom, rather than an application crash. The all-red display is not a recorded actuator-value measurement. There were no reported loop changes. Deployed revisions are unknown and are not required to continue the investigation. The BMC/tweeter path is now the incident focus; the ALPAO mean-removal and optional woofer response-matrix paths are not primary suspects for this event. The deployed tweeter GPU setting remains unverified despite the local configuration selecting GPU 1.

This lowers the incident priority of deliberate-reconfiguration races, size-changing GPU leaks, and startup/schema failures. Unexpected reconnects remain possible but have not been observed. It raises the priority of continuously active feedback coherence, freshness, and numerical conditioning. Static calibration defects can also remain latent until commands grow or enter saturation.

## Assessment

The highest-priority suspects are in the continuously running feedback path:

1. mfilt can accept a modal vector that combines two dmRecon publications.
2. dmRecon queues wakeups for a single mutable result buffer, allowing repeated publication of the same result as new feedback. mfilt treats each observed changed counter as a fresh integrator replacement; publications between its iterations can be skipped.
3. dmRecon can reconstruct a command assembled from two incoming DM frames.

These require no configuration change and are compatible with a rare scheduling interruption after minutes of otherwise normal operation. They establish ways to inject incorrect or stale state, not proof that such an injection will destabilize this particular closed loop. Reconfiguration has additional genuine C++ data races and potential invalid memory accesses. Missing finite-value checks can let one numerical fault persist in mfilt even though its modal limits are enabled.

The existing `m_modevalsMutex` fixes part of the in-process vector handoff. It does not synchronize mfilt, protect input shared memory, bind notifications to particular results, or serialize calibration reloads.

### Interpretation of all-actuator saturation

In the inspected mfilt, per-mode limiting follows both replacement and the WFS increment (`modalfilter.c:1439`, `1462–1474`). With finite values and finite nonnegative limits, each stored modal coefficient is bounded. The observed actuator saturation therefore does not establish that the modal accumulator grew without bound: a sum of individually bounded modes can still exceed the actuator limits. BMC separately clips the actuator command (`bmcCtrl.hpp:1162–1175`); ALPAO removes the mean and clips (`alpaoCtrl.hpp:517–542`). These are source observations, not confirmation of the failed deployment's limits or controller.

The ordinary delta path has the expected subtraction: `makeDelta()` forms the controller's clipped output shape minus the sum of non-delta channels (`dm.hpp:4121–4128`), and dmRecon reconstructs that masked residual with its pseudo-inverse. No unconditional sign reversal or second accumulation was identified in those statements. The remaining questions are whether shape and subtracted channels belong to the same command, and whether the resulting replacement is coherent, correctly calibrated, and timely.

Two control hypotheses need specific offline tests alongside the race repairs:

- **Conditioning and cancellation:** With a poorly conditioned modes matrix, a modest actuator shape can correspond to large cancelling modal coefficients. Combining parts of two modal vectors, or clipping their coefficients independently in mfilt, can destroy that cancellation and produce a large actuator command. Default inversion retains all singular values (F6). The actual modes/mask/inverse are needed to establish whether this applies; no ill-conditioned on-sky calibration has been demonstrated.
- **Timing-dependent dynamics:** Repeated, late, or missed replacements change the feedback recurrence. The existing duplicate-publication model demonstrates lost integrator progress, **not divergent growth or wind-up**. A closed-loop replay with realistic gains, leak, latency, and actuator clipping is needed to connect that defect to the reported symptom.

A pseudo-inverse is also not an actuator-bounds constraint. Even exact projection of a clipped shape back into a modal subspace can exceed an individual actuator limit. For example, with a single mode `A = [2, 1]^T`, `A A+ [1, 1]^T = [1.2, 0.6]^T`. This illustrates why successful inversion does not guarantee an unsaturated reconstructed command. Ideal orthogonal projection is nonexpansive in the Euclidean norm, so this example alone is **not** an instability demonstration.

Incident ranking after the clarification: investigate F1–F3 together with finite-value/conditioning checks from F6 first; treat F4–F5 and lifecycle failures as secondary causes unless existing logs show a reconnect or calibration event. A common rail versus mixed rails would help distinguish an offset/piston-like failure from a growing spatial pattern, without uniquely identifying the cause. That distinction cannot be recovered from the reported all-red display; it requires existing numeric records or an isolated software reproduction.

## Scope and evidence baseline

`hostname` returned `mx11`. No instrument commands, live stream attachments, hardware tests, or application launches were performed. Reads covered local source, local configuration files, and dependency headers. A small deterministic protocol model was run in `/tmp`; it did not execute MagAOX or cacao code.

| Source | Reviewed baseline |
| --- | --- |
| MagAOX | `d6af516404453825b2dc952c5b79f30e9defaee2`, branch `jrmales/dm-syncopation` |
| Last dmRecon change | `70a2e1b4a944f3a3da830c36e312b71bafd1e519`, "Harden dmRecon cross-thread state handoff" |
| Local cacao | `/home/jrmales/Source/cacao-nt`, `390b268895ef2c736a567870687da0407aec0411`, clean |
| Local mxlib | `/home/jrmales/Source/mxlib`, `32c6a0b3d8691bce6c84b36a9690d25d07eaf106`, clean |
| Local milk | `/home/jrmales/Source/milk-xwcl`, `2c4ac9d9ed3efa37ae0a631dc60e0fbd7e8f16fd`; modified ImageStreamIO submodule at clean checkout `21a8ae160f16a934a287ba82d14810043c19f47c` |
| Local configuration | `/opt/MagAOX/config`, `85e20a0e9df058b4fcdcd3ba3474b81794ef9070`, clean |

The installed `eigenLapack.hpp`, `milkImage.hpp`, and `cudaPtr.hpp` match the inspected mxlib checkout. Local `dmtweeter-recon.conf` selects GPU 1; local `dmwoofer-recon.conf` uses CPU defaults and loop 0. These are workstation facts, **not confirmation of the versions or settings used on sky**. The pre-existing untracked `xlog/` directory was not changed.

Source references below use line numbers at these baselines. Main files:

- [dmRecon.hpp](../../../apps/dmRecon/dmRecon.hpp)
- [shmimMonitor.hpp](../../../libMagAOX/app/dev/shmimMonitor.hpp)
- [frameGrabber.hpp](../../../libMagAOX/app/dev/frameGrabber.hpp)
- [cacao modalfilter.c](/home/jrmales/Source/cacao-nt/AOloopControl/modalfilter.c)
- [mxlib eigenLapack.hpp](/home/jrmales/Source/mxlib/include/math/eigenLapack.hpp)
- [mxlib milkImage.hpp](/home/jrmales/Source/mxlib/include/improc/milkImage.hpp)

## Thread and stream path

The modes monitor constructs `m_PInv`, the mask monitor constructs `m_maskIDX`, and the command monitor extracts pixels and computes `m_modevals`. A separate frame-grabber thread publishes those mode amplitudes. INDI callbacks and the application thread also access shared state.

```text
CMmodesDM + actuator mask
          -> mask indices + pseudo-inverse
dmNNdisp_delta
          -> command monitor -> m_modevals + semaphore
          -> frame-grabber thread -> aolN_modevalDMf
          -> mfilt replaces mvalDMc
          -> gain/leak integration -> modal limiting -> next DM command
```

mfilt tests `modevalDMf.cnt0` once per iteration. For a changed counter it replaces `mvalDMc`, then applies `dmval + mvalDMc * multiplier` and modal limits (`modalfilter.c:1427–1475`). This is state replacement, so an old result advertised with a new publication counter is not harmless repeated telemetry.

## Findings

### F1 — High: mfilt can accept a torn feedback vector

**Confirmed protocol gap; steady-state trigger.** `modalfilter.c:1431–1440` reads `cnt0`, records it as consumed, and copies the floats directly into `mvalDMc`. It neither tests `md->write` nor rechecks the counter after copying. The publisher marks `write=1`, copies the vector, increments `cnt0`, then clears `write` and posts ([frameGrabber.hpp:1050](../../../libMagAOX/app/dev/frameGrabber.hpp#L1050)–1093).

Permitted interleaving:

1. mfilt sees completed feedback A and copies its first modes.
2. dmRecon starts overwriting the same shared array with B.
3. mfilt copies the remaining modes from B and immediately integrates the mixed vector.

An alternative is that A's counter is still unconsumed when B is already being written; mfilt sees A's counter and reads B in progress. The local mutex around `loadImageIntoStream()` is invisible to the other process. Semaphore publication does not prevent the next write, and mfilt does not use that semaphore to claim ownership of the array.

**Current action:** Retain F1 as an unresolved protocol gap and add the user-selected passive timing diagnostic. dmRecon publishes independently and then reports whether the exact next observed trigger timestamp followed completion. No mfilt changes or trigger gate are part of this implementation. The historical coordinated-reader proposal below remains a possible future repair, not a prerequisite or an approved next step: copy into a private candidate, validate a complete unchanged publication, and only then replace `mvalDMc` and advance the consumed counter. Such a repair must establish synchronization and memory ordering on both sides; checking `write` once is insufficient, and an uncoordinated two-buffer swap alone does not prevent reuse while a slow reader is copying.

#### Historical mfilt repair proposal: outside the selected implementation

The mfilt-side change can remain localized: allocate one `NBmode`-float scratch vector beside the existing work buffers, replace the acceptance block at `modalfilter.c:1431–1440`, and free the vector during existing cleanup (`2374–2396`). Expect a few dozen functional lines plus a focused test harness; this is a scope estimate, not an implemented patch.

On a changed publication, make one bounded snapshot attempt: check writer state and counter, copy into scratch, recheck them with the agreed memory ordering, and accept only a complete unchanged publication. Advance the consumed counter only on acceptance. On overlap, skip the external replacement for that iteration and continue the ordinary WFS/gain/leak update from the existing integrator state. Do not freeze the loop or wait for dmRecon; count rejected snapshots for diagnosis.

Scratch storage is `4 * NBmode` bytes (about 8 kB for an illustrative 2,000 modes, not a measurement of the failed run). The simplest version adds one private-vector copy per accepted update plus a few metadata reads. A small refinement can avoid that extra copy: the existing integration loop reads its prior state from the accepted scratch vector, or from `mvalDMc` when no candidate was accepted, and writes its result to `mvalDMc`. The scalar operation order can remain unchanged. Neither version needs per-frame allocation, a GPU transfer, a blocking lock, or an unbounded retry. Runtime overhead must be measured in the isolated harness; no microsecond estimate has been established.

The substantive qualification is synchronization correctness, not code volume. The inspected ImageStreamIO metadata declares ordinary `cnt0` and `write` fields, and the current frame-grabber publisher uses ordinary stores. A reader-only check/fence patch must not be described as a proven interprocess protocol without validating the writer side too. That may require a small coordinated publication change or a stronger ownership scheme. F2 duplicate/freshness behavior remains separate even after F1 is repaired.

### F2 — High: queued wakeups can replay feedback and survive disable

**Confirmed logic defect; steady-state trigger.** Each reconstruction sets `m_updated=true` and posts `m_smSemaphore` when enabled ([dmRecon.hpp:1084](../../../apps/dmRecon/dmRecon.hpp#L1084)–1093). There is one result buffer. The consumer accepts each queued token while `m_updated && m_commandReady` is true (`1239–1243`) and copies the current buffer (`1258–1261`). It never consumes/clears the updated state; the only reset is during allocation (`1000`).

A delayed frame-grabber thread can wake with several tokens but only one latest result. It can publish that same result several times, incrementing the external counter each time. If mfilt runs between those publications, each one replaces integrator progress with the same older state. Conversely, intermediate reconstructions disappear. A semaphore count is therefore not a count of retained frames.

The `writeDMf` toggle gates producer posts only (`1086`, `1304–1314`). Already queued work can still publish after disable because neither acquisition nor publication checks the toggle. Queue contents also have no configuration generation. Merely clearing `m_updated` without locking result selection would introduce lost-update races.

**Action:** Use a synchronized latest-result handoff with a monotonically increasing result identity, calibration generation, input identity, timestamp, and publish-once tracking. Coalesce notifications; do not replay them as distinct frames. Freeze the result selected for a publication. Recheck enable/generation at the publication boundary and define whether disabling waits for an in-flight publication to finish. Invalidate queued work on reconfiguration. A bounded queue is an alternative only if every retained result and its age are explicitly managed.

**Related control issue:** mfilt accepts any different feedback counter without source-command correlation or an age limit. Even coherent feedback arriving late can roll back its state. Agree on which applied DM command each replacement represents and how late feedback is handled; do not assume that fixing torn copies alone establishes correct loop timing.

### F3 — High: dmRecon can reconstruct a torn input command

**Confirmed protocol gap; steady-state trigger.** `shmimMonitor` waits for a semaphore and passes a pointer into live shared memory (`shmimMonitor.hpp:817–863`). dmRecon then gathers pixels individually ([dmRecon.hpp:1019](../../../apps/dmRecon/dmRecon.hpp#L1019)–1023). Neither brackets the copy with publication checks or claims reader ownership. The next producer update can overlap the gather.

This can inject a modal vector which represents neither complete input command. A semaphore indicates that a publication happened; it does not freeze that publication for the reader. Modes and mask reads have the same snapshot issue when those streams are updated. The diagnostic `m_modeval` read at `1102` is also unprotected, so `_diff` can compare unrelated generations even when the reconstruction itself is correct.

**Action:** Capture a validated private input snapshot and its publication identity before masking/reconstruction. On overlap, reject or retry with a bounded policy. Record drops. Apply the same rule to calibration snapshots. Inspect the upstream delta calculation separately: `dev::dm::makeDelta()` re-reads live non-delta channels after the controller command (`dm.hpp:4114–4137`), so a coherent delta publication alone does not prove that all upstream contributions describe one command.

### F4 — High: calibration rebuilds and reconfiguration do not have safe ownership

**Confirmed C++ race paths; require reload, reconnect, or interrupted reconfiguration.** Atomic readiness flags publish individual states but do not hold the objects stable throughout their use.

**Mask versus inverse:** Modes allocation can pass the old mask-ready test (`747`), followed by mask allocation clearing readiness (`861`). Both callbacks wait only for `m_commandReady=false` (`779`, `885`). Once the command thread clears it (`925`), the modes thread can read `m_maskIDX.size()` and elements (`802`, `816`, `821–823`) while the mask thread clears and grows that vector (`893`, `904`). This is undefined behavior, with possible stale storage, out-of-bounds access, and inconsistent matrix dimensions or actuator selection.

**Interrupted waits:** The waits at `779` and `885` also end on restart or shutdown, but execution then proceeds into buffer use/mutation without checking that the command thread actually became quiescent. A cancellation path must return instead of proceeding as though the wait succeeded.

**Reused acknowledgment:** `m_fgWaiting` does not identify a generation. The frame-grabber can observe `m_commandReady=true` (`1113`), pause before clearing `m_fgWaiting` (`1120`), and overlap a new command allocation which accepts the old waiting flag (`939`) and resizes `m_modevals` (`950`). The frame-grabber then reads that object's dimensions (`1122–1123`). The copy mutex does not guard either the resize or these configuration reads.

**Restart flags:** Each `shmimMonitor::m_restart` is a plain `bool` (`shmimMonitor.hpp:251`) accessed across monitor threads by dmRecon (`725`, `765–767`, `865`, `876–879`). This remains a data race despite the ready flags being atomic. `frameGrabber::m_reconfig` is already atomic in this baseline.

**Action:** Give the mask, dimensions, inverse, and CPU/GPU buffers one explicit calibration generation. Build candidate calibration state from immutable private snapshots; validate it before committing. Serialize state replacement and require acknowledgments from both command processing and publication for the particular generation being retired. Avoid holding a real-time output lock throughout SVD. Make restart requests synchronized, while preserving requests arriving during restart. Atomic flags alone are not a replacement for object lifetime protection.

### F5 — High under mismatched streams: type, shape, and cube-offset assumptions can access invalid memory

**Confirmed missing checks; occurrence depends on stream contents/configuration.**

- All three input paths cast raw storage to `float` (`775`, `891`, `1022`); the proposed type checks at `756` and `867` are comments. A narrower source type can cause an over-read; other types can silently produce meaningless floats.
- The modes monitor passes `array.raw + cnt1 * width * height * typeSize` (`shmimMonitor.hpp:760–793`, `821–861`), but dmRecon copies an entire modes cube from that pointer (`771–775`). For full-depth reconstruction, any positive `cnt1` exceeds the payload. For truncated reconstruction, the requirement is `cnt1 + selectedDepth <= streamDepth`, and even a bounded offset changes which modes are used. The local calibration loader uses `milk-FITS2shm` (`cacao-calib-apply:109`); its copy path leaves `cnt1` unchanged, and fresh streams initialize it to zero. No normal producer setting it nonzero was found, making this a weaker match to the incident. The failed run's metadata remains unknown. A calibration cube should be read from its defined base, not implicitly treated as one frame of a circular buffer.
- The diagnostic loop bounds itself using `m_modevalMon.rows()` but reads `m_modeval(r,0)` without verifying that the separately opened `aolN_modevalDM` has that many rows (`954`, `1099–1102`). mxlib validates the type on open, but its element accessor does not add range checks (`milkImage.hpp:767–775`); Eigen assertions may catch this in a checked build, while an unchecked build can over-read.
- The optional response-matrix path takes `floor(sqrt(rows))` and maps that squared size, without validating a perfect-square output or matching input columns (`497–498`, `788–800`). Invalid dimensions can assert or misbehave when assertions are disabled. This path is commented out in the local woofer config.

**Existing protection:** The passive frame-grabber already checks output width, height, and depth and waits on a mismatch (`frameGrabber.hpp:958–975`). Do not report an unconditional missing output-size check. It does not compare the attached stream's datatype to `_DATATYPE_FLOAT` before dmRecon copies floats into it.

**Action:** Validate datatype, dimensions, element count, byte capacity, nonempty selected mask, selected mode count, and every mask index before enabling reconstruction/publication. Validate the diagnostic stream independently. Define whole-cube versus frame semantics explicitly. Cover malformed shapes in release-mode sanitizer tests as well as assertion-enabled builds.

### F6 — High consequence: invalid numerical state is published without rejection

**Confirmed validation/error-handling defects; numerical triggers remain unobserved.**

- dmRecon tests only `rv < 0` after `eigenPseudoInverse()` (`833–835`). The inspected mxlib returns any nonzero LAPACK SVD status, including positive failure (`eigenLapack.hpp:1073–1078`). Positive failure can therefore leave an old/empty inverse while dmRecon logs success and marks modes ready (`842–853`). `condition` and `nRejected` may also remain unset on that path.
- No finite checks guard calibration, input commands, reconstructed results, or final publication. The optional response normalization divides by mask size and mode norm without zero/finite checks (`802–805`).
- The default `inverseNumModes=0` retains all singular values. The dependency's mode-count branch takes `1/S(i)` without a zero-singular-value floor (`eigenLapack.hpp:983–999`). A singular or poorly conditioned calibration can generate nonfinite coefficients or strongly amplify a small input inconsistency.
- `condition` is uninitialized at `dmRecon.hpp:827`; the mode-count branch in mxlib compares against its incoming value (`eigenLapack.hpp:992–993`). This compromises the diagnostic and is an indeterminate-value bug; it is not, by itself, evidence that the inverse coefficients are wrong.

If a NaN reaches mfilt, both ordinary `> limit` and `< -limit` comparisons are false (`modalfilter.c:1466–1474`). A NaN can thus survive limiting and integration until valid feedback or a reset replaces it. That is a plausible way for one rare bad input to persist across iterations, including without an application crash.

**Action:** Treat every nonzero inversion status as failure; initialize diagnostics; reject empty/singular/invalid calibration before activation. Check input, inverse, and result finiteness and reject bad feedback before incrementing its publication counter. A defensive finite check at mfilt's acceptance boundary remains deferred under the selected dmRecon-only scope. Define any conditioning or amplitude thresholds with the actual calibration/control requirements, rather than silently changing the science behavior in a safety patch.

### F7 — Medium: ancillary state and stream lifecycle can obscure or interrupt feedback

- The command monitor writes `m_currImageTimestamp` (`1017`), while the frame-grabber reads it (`frameGrabber.hpp:1071`, `1082`, `1086`) without synchronization. A later command's timestamp can be paired with an earlier result, and the concurrent accesses themselves are a C++ data race. Bundle timestamp and vector in the result handoff.
- INDI writes `m_fps` under `m_indiMutex` (`1280–1287`), while `fps()` reads it without that lock (`1211`). Synchronize both sides. This is primarily a telemetry/configuration issue, not the leading integrator explanation.
- `configureAcquisition()` records the output stream inode (`1195`), but the normal frame-grabber publication loop never checks it. Replacement of the backing file can leave dmRecon writing an old mapping while a restarted consumer uses a new stream. Unlink/recreate alone does not make the old mapping a dangling pointer; the expected failure is disconnected/stale feedback. The diagnostic `m_modeval` mapping is likewise reopened only during command allocation.
- `milkImage::open/create/post` can throw; the dmRecon callback and ordinary monitor loop do not catch these failures. A missing diagnostic stream on reconnect can terminate the process rather than enter a controlled retry. This is relevant if "blow up" meant a crash, and also means diagnostic-only failures can stop the feedback process.

**Action:** Define reconnect behavior, invalidate feedback generations on mapping changes, and handle diagnostic connection failures deliberately. Keep ancillary diagnostic availability separate from acceptance of a valid reconstruction where practical.

### F8 — Additional correctness and coverage issues

- **Mask storage order:** `m_maskIDX` is built with row outermost, column innermost, and a sequential index (`898–908`), while `mx::improc::eigenImage` uses Eigen's default column-major storage. For a 3x3 mask with only `(0,1)` active, the stored index is 1 rather than 3. A symmetric mask hides this; an asymmetric square mask selects its transpose. Both modes and command use the same wrong selection, so this is not automatically an inconsistent matrix multiply, but it selects the wrong actuators and can change conditioning. Iterate raw mask storage or compute the actual storage index.
- **Non-CUDA build:** The `#else` calculation at `1079` lacks a semicolon and does not use `m_modevalsMutex`. CPU execution inside a CUDA-enabled build does have the mutex (`1030–1031`). Repair and exercise the separate non-CUDA build; do not claim that all CPU operation is currently unlocked.
- **Tests:** The former `apps/dmRecon/tests/dmRecon_tests.hpp` (removed during implementation) was an unrelated template which includes `../t2wOffloader.hpp`. It supplied no regression coverage of dmRecon's handoff, calibration, or bounds behavior.
- **GPU allocation leak on size changes:** The inspected `mxlib/include/math/cuda/cudaPtr.hpp:251–265` assigns a new `cudaMalloc` allocation to its stored pointer when the size changes without freeing the previous allocation. dmRecon reaches this through inverse upload and vector resizing (`973`, `981`, `988`). Repeated size-changing calibration reloads can exhaust device memory; unchanged steady-state processing does not allocate through this path. Fix and test this in mxlib. The ordinary upload/download path uses synchronous `cudaMemcpy`; no asynchronous host-buffer lifetime defect was identified there.
- **mfilt startup initialization:** Local `modalfilter.c:880` has `for(uint32_t mi; mi < NBmode; mi++)`, with an uninitialized index when clearing `mvalDMc`. Correct this independently. Initial external-state substitution commonly overwrites the state, and this startup defect is a weaker explanation for a failure after minutes of stable operation.

## Verification performed during the original audit

The original review traced the producer, both dmRecon worker stages, mfilt acceptance, and the dependency behavior above. At that stage no full application build, runtime reproduction, ASan, TSan, or GPU sanitizer run was performed. The 2026-11-15 implementation validation is recorded separately above; the configured application was never launched.

A deterministic Python statement-interleaving model at `/tmp/dmrecon_mfilt_protocol_repro.py` ran successfully with assertions for these three cases:

| Model | Observed result |
| --- | --- |
| Read the first mode of A, overwrite the shared vector with B, read remaining modes | Accepted vector `[1, 2, 2, 2]`, despite complete vectors being all 1s or all 2s |
| Produce 9 then 10 before consuming two wakeups; mfilt adds 1 between publications | Integrator states `[11, 11]`; one replacement followed by two increments would yield `[11, 12]` |
| Disable `writeDMf` while an accepted token remains queued | Publication still passes the current acquisition predicate |

These are executable demonstrations of the allowed algorithmic schedules. They are **not** executions of the application, sanitizer reports, failure-frequency measurements, or evidence that the on-sky event followed those schedules. The schedules and expected outputs in this table preserve the useful evidence if the temporary script is removed.

## Proposed implementation and test order

1. **Use the clarified failure baseline.** The reported symptom is apparent integrator wind-up on the tweeter, inferred from an all-red display after several minutes, with no loop changes and unknown revisions. Rail direction is unknown because the display does not distinguish it. Continue without requiring the user to recall revisions or infer rail direction. If existing deployment records become available, check for `70a2e1b4`; older code additionally lacks the current vector-copy mutex and atomic ready flags. Use available records to establish CPU/GPU choice and first growth/jump, and numeric actuator values if present. Examine existing logs for unexpected reconnects rather than presuming a calibration change.
2. **Fix dmRecon state transfer and observe its timing.** Address F2–F3 through private snapshots, unique identities, generations, timestamps, publish-once semantics, and a defined disable boundary. Add finite rejection from F6 before publication. For F1, implement passive semaphore observation and post-publication exact-successor diagnostics with late/unverified warnings. Leave mfilt and its trigger path unchanged; the underlying reader/writer overlap remains possible.
3. **Fix calibration ownership and schema validation.** Implement F4–F5, inversion failure handling, and mask storage-order correction. Keep failed candidate calibration from replacing the last valid generation; explicitly inhibit feedback whenever consistency with the live streams cannot be established.
4. **Complete lifecycle/build work.** Address F7, compile the non-CUDA branch, and replace the placeholder tests with an isolated harness following repository application-unit-test and Doxygen conventions.
5. **Validate timing and control behavior offline.** Replay representative commands/calibration through the corrected pipeline. Check modal-to-actuator-to-modal consistency, retained singular values, finite output, and reconstructed actuator shapes before and after clipping. Include fixed non-delta offsets and deliberately mismatched offset generations. Measure reconstruction-to-consumption age and variability, rejection counts, and integrator response under injected stalls. Model the reported all-actuator saturation with realistic WFS feedback, gains, leak, and limits; a queue-only model cannot establish wind-up. Finite coherent data may still be too old for stable substitution, so this step remains necessary after memory/race fixes.

Reuse the existing `shmimMonitor`, `frameGrabber`, and mxlib facilities. The repository has related bounded-copy patterns in [dmSyncopation::snapshot](../../../apps/dmSyncopation/dmSyncopation.hpp#L2640) and [synco::readRecord](../../../libMagAOX/app/dev/dmSyncopationProducer.hpp#L49). They are references for staging and repeated validation, not a proof of portable interprocess synchronization. If both feedback readers need the same validated-copy facility, factor a small common helper after its ownership and memory-order contract is settled; avoid copying app-specific synchronization machinery wholesale.

Planned workstation-only validation:

| Test | Required outcome |
| --- | --- |
| Delay the trigger callback until after publication while its source timestamp precedes completion | Report `late` from source time; callback order must not produce a false success |
| Observe an exact successor after completion, including a callback that precedes the completion record | Classify only after matching source/trigger identities and clock evidence |
| Drop a successor observation, change inode/counter generation, or inject a clock anomaly | Report unverified; never infer success from a later counter or callback receipt time |
| Pause mfilt halfway through copying; publish another vector | Preserve the original F1 reproducer as evidence of an unresolved gap; this passive implementation must not claim to prevent it |
| Pause DM input gather across a producer update | Mixed input is rejected before reconstruction/publication |
| Stall publication for many input frames, then resume | Latest-result policy publishes each selected identity at most once; no old-token replay |
| Disable or reconfigure with queued/in-flight output | Defined disable boundary is honored; no retired-generation feedback is accepted |
| Overlap mask rebuild, inverse build, command processing, and FG configure using barriers | No TSan race; no ASan/UBSan violation; one internally consistent calibration per command |
| Wrong types, short vectors, nonzero cube `cnt1`, unequal mode counts, empty/asymmetric masks, bad response dimensions | Deterministic rejection or correct documented cube semantics; no payload overrun in release builds |
| Inject positive SVD failure, zero singular values, NaN/Inf, GPU errors | Invalid state is never published or accepted as replacement; errors are observable |
| Recreate producer/consumer streams in an isolated namespace | Old mappings/generations are retired and reconnect behavior is explicit |
| CPU with/without CUDA build support, plus GPU execution in an isolated harness | Equivalent valid reconstructions and consistent failure handling |
| Actual calibration round trips, near-null modes, static offsets, saturation and per-mode limiting | Quantify reconstruction bias, conditioning, and loss of modal cancellation; distinguish finite actuator saturation from nonfinite data |
| Closed-loop replay with unchanged settings and injected late/duplicate/torn feedback | Determine whether the defects can produce sustained all-actuator saturation, rather than merely dropped progress |
| Sustained replay longer than the reported failure interval, with scheduling stalls | No unexplained result identities, stale replacement, or nonfinite state; latency distribution recorded |

TSan covers in-process races in the harness, not synchronization between separate processes or CUDA internals. Use deterministic barriers and process-level fault injection for those boundaries. Keep all synthetic stream names and runtime paths isolated from instrument streams.

## Evidence still needed from the failed run

- Deployed tweeter configuration, GPU setting, and whether `writeDMf` remained enabled throughout. The affected DM is established as the tweeter. Revisions are unknown; recover them from existing records only if available.
- Existing numeric records, if any, showing actuator values and whether the first anomaly was a sudden jump or gradual growth. Apparent wind-up/all-actuator saturation is the reported symptom, based on an all-red display; numerical wind-up and rail direction have not been measured. Do not ask the user to infer rail direction from that display.
- Existing evidence of unexpected stream restarts/recreation. No loop changes were reported; do not presume a deliberate mode/mask update.
- If recordings already exist: DM delta commands, `_modevalDMf_mon`, `_modevalDMf_diff`, mfilt modal output, publication counters/times, and inversion logs. Current `_diff` and timestamps have the limitations described above.

Any interrogation of the real instrument must be performed by the user. This document does not request that an unstable closed loop be rerun to collect evidence.

## Changes from this investigation

The original audit added this planning document as a documentation-only change. On 2026-11-15 the user authorized implementation within dmRecon, selected passive timing diagnostics, and explicitly rejected gating mfilt. The application/helper/test changes and their validation are recorded above. Changed paths are `apps/dmRecon/dmRecon.hpp`, `apps/dmRecon/dmReconMath.hpp`, `apps/dmRecon/dmReconTiming.hpp`, `apps/dmRecon/Makefile`, the three `apps/dmRecon/tests/*_test.cpp` files, deletion of the unrelated `dmRecon_tests.hpp`, `tests/tests.list`, `tests/groups.dox`, and this plan. The implementation and this engineering record are committed together on the review branch. Cacao/mfilt and its trigger configuration remain unchanged. On-sky causality, F1 protection, control-law stability under late feedback, and instrument latency measurements remain open.

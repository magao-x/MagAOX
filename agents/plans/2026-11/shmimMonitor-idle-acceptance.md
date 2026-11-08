# NN / PO4AO / collector acceptance and PR preparation

Prepared 2026-11-08. This is an operator-run test plan, not a deployment approval or a record of completed instrument tests. All agent work stays on the workstation; no instrument SSH, INDI connection/tunnel, utilities or hardware actions are authorized.

## Current release gate

The review fixes are committed, and the dedicated feature branch integrates `dev` at `16138ab7b5b90bae028fb09aeb2c73e3f391844a`. See the [integration record](shmimMonitor-idle-apps.md#current-dev-integration-decision-2026-11-08) for conflict resolutions and workstation evidence. PR #418 is already included by ancestry. Nothing has been pushed or deployed by this integration.

The user approved unchanged upstream READY behavior. NN Disable, whole-app PO4AO Stop and collector Stop/completion acknowledge app work completion and an asynchronous READY request. They do not acknowledge that the mapping has closed. Observe reader release and absence of semaphore waits separately during qualification. The shared monitor, base app and generic GUI publisher match dev; PO4AO telemetry uses new code 20908 without changing upstream codes 20906/20907.

## 1. Freeze and validate a candidate off instrument

1. Freeze the integrated feature-branch commit and check for newer dev changes before final qualification. Preserve PR #418 plus GUI, saved pairs, independent controls and collector improvements. Do not use `nnreconstructor-speedups`.
2. Build libMagAOX and regenerated logger definitions, all affected native apps, log readers and the GUI against the actual intended Linux/milk/ImageStreamIO/CUDA/TensorRT versions. Include relevant common-app and upstream monitor-event tests, not only these app fixtures. Rerun the app, GUI, trainer, collector and telemetry suites after integration.
3. Run a real ImageStreamIO synthetic producer in an isolated environment. Exercise missing-source startup, attachment cancellation, rapid stop/start, stalled producer and stream recreation. Observe actual semaphore reader ownership and monitor-thread waits while the producer continues. CPU reduction alone is not proof of detachment.
4. On an isolated GPU host, test repeated NN Disable/Enable, invalid engine and load failure, disable during load/inference, final zero and no late residency. Verify owned resources are released; a retained CUDA context may still use VRAM. Decode real PO4AO telemetry, including idle and model-transition windows.
5. Push the exact candidate and configuration revisions to the approved remotes before on-sky use. Record commit, dirty status, dependency versions and model/bank hashes with the results. Keep an identified known-good software/configuration set for rollback. Do not change source while qualifying a candidate; a functional correction requires retesting affected gates.

## 2. Operator setup and deployment on the internal source

Use the normal [daily startup](https://magao-x.org/docs/handbook/operating/daily_startup.html) and alignment procedure. Establish the internal source, valid WFS darks, compatible calibration and visible logs/INDI before these tests. Begin with the HOWFS loop open, learning paused and the collector idle. Loop closure and physical configuration remain operator actions.

Record the currently working route, writer, calibration, WFS rate/exposure/gain, modulation, model identities, output geometry and timing baseline. Establish model compatibility from provenance: input normalization, pupil geometry, modal basis/order/sign/units and output dimensions. Matching TensorRT dimensions alone is insufficient.

The native HOWFS operations are on RTC/exao2. The inspected root is `/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir`; confirm it and the installed utilities/hooks match the [source evidence](shmimMonitor-idle-apps-evidence/README.md). Confirm installed FPS/process-info layouts and actual stream names. The GUI runs in the existing operator GUI environment (previously the IOC container); verify its INDI endpoint exposes both native devices and the optional trainer when used.

Build before replacing any running executable. This is a full native update: libMagAOX/logger generation, po4ao, nnReconstructor, dlDataCollection, the matching GUI, and po4aoTrainer when testing managed training. The older trainer-only deployment shortcut does not apply. Use the established deployment procedure, preserve installed binaries/configuration, and check every application startup log for rejected settings. NN configurations must omit obsolete `explicit_tt`, `use_fp16` and `dataDirs`; see its README. Select a deliberate startup enable state that preserves exclusive writer ownership.

Verify these capabilities before pressing Start/Stop:

| Device/readback | Required evidence |
| --- | --- |
| PO4AO `run_enable`, `run_status` | Native controls present; phase, route, error, owner and verified fields readable |
| NN `routing_status` | Owner, source, phase, instance, sequence and `released_owner` present and fresh |
| NN `model_status` | Host-loaded, resident and active are distinguished |
| PO4AO `model_status`, `collection_status`, `runtime_stats` | Current model/collection state and advancing runtime heartbeat |
| Actual CACAO processes/streams | Verified writer identity, live input and downstream counters; property presence alone is insufficient |

If a required readback, model contract, source version or exclusive-writer condition is unavailable, stop this stage and resolve it before a route change. Do not start a second notebook/exporter alongside managed training.

## 3. Internal-source sequence and pass criteria

Start with the optional trainer stopped so native routing is tested independently. Keep the HOWFS loop open throughout routing tests, as required by the [CACAO calibration procedure](https://magao-x.org/docs/handbook/operating/cacao.html). Whole-app Start/Stop applies default calibration; it is not a closed-loop seamless-handover control.

| Test | Action | Required result |
| --- | --- | --- |
| NN lifecycle | With a safe operator-managed output arrangement, Enable, then Disable NN while WFS frames continue; repeat Enable | Off only after in-flight work, final zero and owned GPU cleanup finish; READY requested, `resident` clears, `loaded` stays; independently verify eventual input detachment; Enable recreates resources and waits for fresh inference |
| NN/CACAO handover | Use compound Switch to CNN / Switch to CACAO, then selected-model Set reconstructor | Exactly one actual writer after completion; no simultaneous NN/CACAO publication; requested/loaded/resident/active states agree with observations |
| PO4AO Stop | Pause learning; note selected policy and actual writer; press Stop PO4AO | Busy then stopped/verified; route is `aol1_modevalWFS`; actual mfilt input inode/counters prove consumption; prior writer restored; only then app work drains and READY is requested; independently verify eventual input detachment |
| PO4AO Start | Press Start PO4AO | Fresh PO4AO output exists before routing; calibration-driven new writer is observed before NN restoration; mfilt consumes `po4ao_output`; running/verified only after completion |
| Both writer owners | Repeat Start/Stop with CACAO initially owning the reconstruction output, then NN initially owning it | Each request restores its own captured writer; modulation preference is preserved by routing; no writer overlap |
| Policy versus app | Toggle Enable/Disable policy separately, then Stop/Start the app | Policy toggle selects policy/fallback, without claiming whole-app detachment; app Stop requests READY and retains policy selection; Start never resumes learning |
| Acknowledgements | Check visible reset/reload gates and normal Busy → terminal progression | No permanently disabled button after an explicit rejection; unresolved timeout remains unknown and is not automatically retried |
| Idle cost | Leave source frames running with NN disabled, PO4AO stopped and collector idle | Their input reader slots are released and monitor threads no longer wake on each source frame; record rate/counter and thread-wait evidence, not CPU usage alone |

For NN low-level Disable, do not assume it automatically transfers ownership to CACAO; that is the compound handover's job. Do not issue independent routing/writer commands while whole-app routing is Busy.

Record GPU memory before Enable, while active, after completed Disable and after re-enable, with process identity and timestamps. Repeat enough cycles to distinguish a one-time CUDA context allocation from accumulating owned resources; ten cycles is a proposed test size, not an established instrument qualification threshold.

Once native tests pass, enable the optional trainer and test the saved-pair workflow:

1. Save an acknowledged policy/dynamics pair and record its identity.
2. Set that pair and verify the restored model pair, unchanged intended policy-enable state, paused learning and no unsolicited recollection/noise.
3. Explicitly recollect warmup when appropriate and wait for completion, then explicitly resume learning.
4. Stop/Start PO4AO; verify policy selection survives and learning stays paused.
5. Pause before a controlled trainer restart; verify saved-pair availability and truthful recovery state. Exercise Set/Save with the app stopped; native Start/Stop must remain functional with the trainer unavailable.

Test dlDataCollection separately, following its [quickstart](../../../apps/dlDataCollection/QUICKSTART.md), with a previously approved bank and an exclusive initially zero DM channel. This actively applies bank commands; it is not a passive logging test. Run a short collection to completion and a second to explicit Stop. Require terminal status, completed versus partial file markers, correct image/command correspondence and observing metadata, owned-channel cleanup, READY and input detachment. Do not change ND/camera configuration until collection is terminal. Use synthetic streams for disruptive reconnect/failure tests first; do not inject live DM faults merely to exercise code.

Decode recorded `TELEM_PO4AO` with the candidate logger tools. Compare callback percentiles, fresh sample count/time, exact counter deltas, model ID and backend with INDI. Idle must be marked stale/empty rather than displaying fresh healthy timings. A transition window must be labelled mixed when appropriate. Callback latency does not establish end-to-end WFS-to-DM latency.

## 4. Closed-loop qualification and recovery

After open-loop functionality passes, use the normal operator procedure to establish the desired loop settings and close the loop. Calibration side effects remain part of that procedure; this software does not restore gains/leakage. First qualify the conventional baseline, then NN and PO4AO, using approved models and initial control settings. Open the loop again before each whole-app PO4AO route change.

A flat internal source is only a smoke test. Exercise representative approved disturbance/turbulence, flux, frame rate and operational load, including training, export, data writing and normal downstream consumers. Compare residuals/PSF, command excursions/clipping, dropped frames/deadlines, end-to-end timing where available, resource usage and recovery with the conventional baseline. Define acceptable limits with the operator before each comparison; none have been established by the workstation tests. A 30–60 minute representative-load run is a proposed starting duration, not a proof of reliability or a validated requirement.

Run delayed FIFO execution, missing readbacks, stale/wrong-token recovery, NN restart and failed calibration cases in the isolated replica first. On the instrument, validate the approved recovery procedure without introducing uncontrolled queued commands. A failed Stop must not park PO4AO while downstream may use it. Keep its output available, identify the retained owner and actual route, settle queued work independently, then release the matching NN reservation and verify `released_owner` before retry. A timeout is not permission to kill PO4AO or issue a new owner token. See the [implementation plan](shmimMonitor-idle-apps.md) for detailed recovery constraints.

Stop progression if ownership or consumption is unknown, a route remains unverified, owned GPU memory accumulates, idle consumers retain frame waits, or operating performance violates the predeclared limits. Follow the instrument's normal loop-opening/recovery procedure; preserve evidence before restoring the known-good installation.

## 5. What qualifies on-sky use

Passing this full sequence on the internal source, including relevant disturbances, load and recovery, can qualify the exact configuration for a supervised on-sky commissioning trial. It does not establish routine science readiness across different stars, flux levels, seeing, modulation or calibrations. First on-sky use should repeat conventional-baseline comparisons, introduce one controller change at a time and retain a verified return path. Learning remains an explicit operator choice.

The requested app features are implemented, but upstream integration (including the event-code collision) is still required. Real dependency/ABI/GPU/ImageStreamIO validation may identify further corrections; there is no evidence yet that those gates pass. No additional speculative control framework or automatic gain management is proposed. For on-sky use, commit and push code and configurations, record their identities with data, and satisfy instrument-owner approval, as required by the [development policy](https://magao-x.org/docs/handbook/software/hacking.html).

Suggested result record for every step: candidate/config/model IDs, starting loop/route/writer state, requested action, observed terminal state and elapsed time, actual input/consumer counters, GPU/resource measurements, telemetry/log location, pass/fail and operator initials. Retest affected steps after any source/configuration correction.

## 6. PR #418 integration

Verified on GitHub 2026-11-08: [PR #418](https://github.com/magao-x/MagAOX/pull/418) is open, unmerged, targets `dev`, and has head `4ff2634124a5a6fcf428e22ce2807dc31b21b823` on `rlandman/po4ao-pr`. GitHub reports it non-mergeable. Its discussion fetch returned no comments. That head is an ancestor of this branch: all seven PR commits are already included, with 41 subsequent committed changes plus the current working fixes. Do not cherry-pick it again.

Recommended route: update #418 once the candidate is finalized. Preserve its history, broaden its title/description to cover the GUI, saved pairs, native NN/CACAO controls, collector changes, idle lifecycle and telemetry, and keep base `dev`. After integration and validation, its head branch can be advanced normally to the combined candidate if the ancestry check still passes; no force push should be needed. Recheck for other contributors' intervening commits before updating. If maintainers prefer to keep #418 narrowly scoped, use an explicitly coordinated stacked or superseding PR instead of leaving duplicate independent PRs to merge the same changes.

Before marking ready: commit the functional fixes with their plan, then documentation; resolve current `dev`, rerun tests, include operator results for the exact candidate and all remaining limitations, run CI, and obtain the shared-code/Python reviews required by project policy (Jared for libMagAOX, Joseph for Python/provisioning). Prior instrument results in #418 concern its older revision and do not validate this candidate. Publication or merge is not deployment. No PR metadata, branch, comment or reviewer request was changed while preparing this plan.

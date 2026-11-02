# Response to the independent NN / PO4AO review

Date: 2026-11-02. Branch: `rlandman/nn-po4ao-gui`.
Reviewed HEAD: `4790c3c8c4a6ec1ed24b9278721b596dc8865a01`.
Baseline: `88271a3e6ec3c3add36a4983d806f05501108f20`.

The review is substantively correct. All eight findings are accepted and addressed,
as are the additional artifact, diagnostic, mapping and progress gaps. There is no
strong disagreement with the proposed architecture or corrections. The earlier
statement that the branch was ready for an operator deployment trial was premature;
the reviewed revision had unresolved control-coordination bugs despite passing its
tests. The fixes below provide stronger offline evidence, not target acceptance.

One wording qualification: the original Reset timeout produced Alert rather than
a successful Ok acknowledgement. That did not mitigate the real problem: runtime
invalidation could be skipped while the managed session appeared idle.

## Findings and corrections

| Finding | Implemented correction | Regression evidence |
| --- | --- | --- |
| 1. Reset can skip runtime invalidation | Invalidate managed eligibility at admission; send fallback and `reset_model` before waiting for frame completion or owned-process cleanup. Persist a reset barrier across timeout/restart. Ordinary training/Enable requests remain unavailable until the intended runtime acknowledges completion; Reset can be retried. | Reset without frames, supervisor restart, eventual frame completion, then Enable without reloading the old engine. |
| 2. External Reset/Disable | C++ publishes process identity, requested control target and existing request/applied generations. The companion reconciles external Reset/runtime restart into session invalidation and owned-worker termination; an external Off transition pauses the owned worker. Publication/Enable admission checks the observed runtime epoch. | External INDI Reset/Disable and runtime restart using real PurePyINDI2 message/cache classes. |
| 3. Cancelled reload loses the exact policy reference | Persist pending publication separately from retained eligibility. Send the exact candidate identity to the worker before requesting its load. Cancellation/timeout retains the load ticket and numerical reference; Pause/Disable cannot report Ok until an issued load settles. Only a confirmed matching activation promotes eligibility. Reset invalidates old pending work. | Delayed load through Runtime, Supervisor and a real CPU TrainingSession, Disable, eventual activation, Resume and a subsequent numerical step; timeout and Reset races. |
| 4. Old cache looks current | Track received definitions separately from the client's retained cache. Invalidate transport/device/property lifetimes, reject undefined/expired observations and request fresh definitions periodically. Capture completed property snapshots. C++ publishes one model-status vector with advancing app-logic sequence; a frozen sequence expires even if definitions are re-served. GUI requires an advancing companion heartbeat and expires it after five seconds. | Actual retained PurePyINDI2 cache across reconnect/deletion; silence versus unchanged fresh definitions; frozen app-logic heartbeat; Qt expiry/recovery and repeated-heartbeat rejection. |
| 5. NN fault blocks recovery | Treat Off plus empty active filename and the donor's disabled/fault phase as confirmed quiescence. Activation still requires healthy active output. Distinguish verified `stopped` from unknown/conflicting/transitional writers; expose explicit Use CACAO recovery and replacement-model Enable. | Donor fault/Alert readback permits CACAO or NN replacement; unverified writer exit still prevents NN enable; Qt stopped-state recovery. |
| 6. Same NN/model replacement changes unrelated state | Return immediately when the selected filename is already active. NN-to-NN replacement disables/loads/enables only NN. Linked modulation applies during an actual CACAO/NN handover. | Same-model no writes and replacement with zero CACAO/modulation writes, with linking both enabled and disabled. |
| 7. Disable unavailable while busy | Give Disable its own availability rule during publication/recollection, while keeping Enable serialized and Reset distinct. | Qt publication/recollection cases send Off while preserving access to independent reconstruction. |
| 8. Invalid model settings accepted | Validate channels/layers >= 2, activation slope in (0,1), and normalization floor fraction in (0,1] before accepting the revision. | Each invalid edit retains the preceding valid configuration and reports its error. |

An externally loaded unknown policy is not adopted as a numerical reference. The
owned session pauses with an actionable error; restoring its eligible policy or
Reset resolves that mismatch. This is numerical identity reconciliation, not
notebook ownership enforcement. No notebook process is discovered or controlled.

## Additional review gaps

- Candidate verification reads the NPZ with pickle disabled, matches model hash,
  session/generation/config identity to the engine sidecar, and checks finite policy
  and dynamics state. Empty, mismatched and nonfinite snapshots are rejected.
- The worker appends publication issuance and terminal outcome to metrics JSONL and
  rewrites the matching diagnostic snapshot. For a given session/iteration, the last
  JSONL record is authoritative. Real subprocess tests cover loaded and failed
  outcomes while paused, in addition to cancellation of a running export child.
- `unmodulated_state` and `modulated_state` are configured app mappings (defaults 3/4),
  checked finite and distinct. The agreed CACAO preset stays 2000 Hz, radius 3.
- The compact panel displays collection frames/target while collecting.

## Validation and scope

- Python: **107 tests and 15 subtests passed**, using the actual installed local
  xconf/PurePyINDI2 framework. Local subprocess tests use normal process visibility;
  they create and terminate only their owned test children.
- Qt widget: **151 assertions in 14 cases passed**, using the production queued
  publisher with a socket-free transport. Launcher rebuilt with local Qt 5.15.15.
  The local build omits unused mxlib pkg-config linkage by qmake command-line override;
  the production project and shared build rules are unchanged.
- Existing publisher suite: **95 assertions in 6 cases passed**.
- Existing PO4AO CPU suite: **259 assertions in 27 cases passed**. These exercise
  production frame/replay methods with substituted SDK/output boundaries, not the
  new full-app INDI status code. This review changes status publication only on the
  C++ side; it does not alter the frame/replay or NN inference algorithms.
- Wheel rebuilt and installed into a temporary local target. All 18 packaged Python
  modules match the current source byte-for-byte; installed supervisor imports work
  from an unrelated directory without importing torch or SHM bindings.
- Black, clang-format and `git diff --check` applied. The explicit `//mutex scope`
  spelling from AGENTS.md is retained after clang-format's comment-spacing pass.
- NN, dlDataCollection and AGENTS.md still match the recorded baseline exactly.
  No excluded speedups, shared-framework changes, loop interlocks, notebook locks,
  remote commands, instrument changes or deployment were introduced.

The original numerical fixture/parity tests remain passing. They cover the tested
synthetic CPU iteration; they do not establish GPU or closed-loop equivalence. Local
Qt builds retain existing SDK/INDI/publisher warnings; Python plots retain existing
font/deprecation warnings.

## Handoff and remaining acceptance

The corrected branch is ready for another independent code review and the documented
operator-run target validation. It is **not established ready for control-capable
instrument use**. Full C++ CUDA/TensorRT compilation (including the new status vector),
real GPU export/reload, actual INDI/FPS/reconnect behavior and timing remain unverified.
Run the acceptance matrix in [DEPLOYMENT.md](../../../apps/po4aoTrainer/DEPLOYMENT.md),
including absent-frame Reset, external Reset/Disable, delayed activation and stale
status, before control-capable use. Target operations remain the operator's task.

A timed-out Reset remains pending until readback/frames confirm it; retry Reset if
the runtime has restarted during that pending request. An unresolved load across a
supervisor restart requires Reset because its in-memory numerical session/ticket was
lost. Old artifacts remain available for analysis but do not restore eligibility.
Legacy saved eligibility without runtime identity also requires Reset, because it
cannot establish whether an external reset occurred while the companion was absent.
Regression tests also verify that external Disable supersedes an Enable/load wait.

Affected implementation files: `apps/po4ao/po4ao.hpp`, `po4ao_impl.hpp`;
`apps/po4aoTrainer/configs/po4aoTrainer.conf`;
`apps/po4aoTrainer/xapp/po4aoTrainer/{app,artifacts,configuration,observations,supervisor,switching,worker}.py`;
`apps/po4aoTrainer/tests/{test_supervisor,test_review_regressions,test_worker_process}.py`;
`gui/widgets/learningCtrl/learningCtrl.hpp` and its test. Related requirements,
implementation record, app/GUI READMEs and deployment guidance are updated with this
response. The existing supervisor/XDevice, worker process, Qt widgets, publisher,
installation rules and C++ `updateIfChanged` helper are reused.

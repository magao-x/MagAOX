# nnReconstructor app-local implementation record

Scope: this directory only. The pre-existing model-testing edits and CUDA_GRAPH.md are untouched.
No instrument connection, command, installation or dependency upgrade is part of this work.

## Reviewable steps

1. Correctness: keep the preprocessing and executeV2 path; validate contracts, check failures,
   own allocations until shutdown, and hand off a protected coefficients/timestamp snapshot.
2. Controls: configured models, separate enable, one zero publication on disable, and explicit
   model reload after faults. No GUI, automatic NN/MVM handover or cached-state loop interlock.
3. Full-file documentation pass, followed by formatting separately.
4. Performance changes deferred until a supported-GPU parity/latency baseline is available.

Existing automatic startup is preserved; a start-disabled option will be added. Model loading
is allowed only after a disabled transition has completed. Selection never enables publication.
The operator is responsible for opening the loop and ensuring exclusive stream-writer ownership.

## Correctness validation

The app-local CPU suite compiles the actual app methods with SDK and application-boundary doubles.
It is not a copy of the production handoff/state logic. The doubles do not validate the real
MagAOX base classes, TensorRT ABI, GPU execution or live shared-memory behavior.

Initial eight cases pass under AddressSanitizer and UndefinedBehaviorSanitizer (258 assertions):
concurrent handoff, timestamp pairing, reconnect buffer reuse, invalid source/tensor contracts,
allocation/copy/inference failures, idempotent cleanup, preprocessing tensor values and shutdown wakeup.

A supported Linux/CUDA/TensorRT build and operator-run end-to-end numerical parity, latency,
shared-memory replacement and downstream-control tests remain required before deployment.
No performance improvement is claimed.

## Runtime-control decisions and limits

The user selected: preserve automatic startup, optionally start disabled, publish zeros once on
an explicit disable transition, select models only while disabled, and require explicit model
reload after runtime faults. Start-disabled is passive (no unsolicited zero); repeated Off after
completed disable is idempotent. Model filenames are explicit configuration choices with readable
sanitized INDI identifiers. Successful model loading never enables reconstruction.

The loader runs off the app logic thread so timeout/error readback remains responsive. Timeouts do
not pretend to cancel native CUDA calls. A timed-out candidate is rejected before installation;
if cleanup stalls after installation, activation remains blocked and the status reports incomplete
loading. Shutdown joins the loader and data-path threads before freeing resources. An indefinitely
stuck driver call can still require an operator-supervised process restart.

Disable closes inference admission, discards unclaimed results, waits for admitted inference and
claimed publication, and sends one zero vector through the existing framegrabber. Off acknowledgement
occurs in its post-publication hook. Shutdown cancels an unclaimed zero rather than acknowledging it.
Faults independently stop admission and invalidate pending results without sending unsolicited zeros.
A disable already requested still completes its zero, but the fault remains until explicit reload.

The base hook establishes completion of bytes/counter publication. The unchanged base discards the
ImageStreamIO semaphore-post return code: it cannot establish successful notification or downstream
consumption. Zero modal measurement is not proof of zero DM command or an open loop.

External output metadata is checked before claiming publication; file identity is checked at most
once per second on the publisher. This adds a periodic filesystem call whose timing cost needs an
operator benchmark. It detects replacement but cannot make it atomic. Disable before controlled
stream replacement. Missing/incompatible output leaves disable incomplete, with an explicit timeout.
Reattachment alone does not clear a fault. WFS reconnect reuses allocations and invalidates pending
results. Re-enable/reconnect wait across a newly observed WFS counter boundary; normal frame handling
and the latest-result handoff are otherwise retained.

Initial estimates omitted about 300 lines of CPU-only SDK/app-boundary test support. That support is
app-local and does not replace production logic. Functional changes stay in the existing app header;
no general framework, shared-base changes or performance rewrite was introduced.

## Final local verification

The completed CPU suite contains 24 cases and 391 assertions, including transition timestamp failure,
late-load cancellation, output identity/location rejection, and failure during disable. The production
preprocessing is tested in both host I/O precisions; the SDK double does not establish GPU parity.
Run the normal, AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer targets listed in README.md.
The final formatting pass uses clang-format 18.1.8 with the repository style; the formatter is an
ignored app-local tool, not a runtime/build dependency change. Functional, documentation and formatting
steps are kept separately in Git. Existing unrelated changes are not included.

## Pre-test correction: production optimization flags (2026-10-19)

The original CPU/sanitizer targets omitted the default production `-O3 -ffast-math` flags.
With those flags, local tests reproduced acceptance of non-finite coefficients and configuration
values: `std::isfinite()` could be optimized away. The app now checks IEEE exponent bits through
const references, without changing compiler flags or reconstruction arithmetic. Passing these
values by value can itself carry the compiler's finite-value assumption, so references matter.

Commit ad7f3158 adds the correction and an `optimized` test target. The expanded suite passes
26 cases / 417 assertions in ordinary, optimized, and AddressSanitizer/UndefinedBehaviorSanitizer
builds on the local Clang compiler. Non-finite output coverage now uses explicit storage bits in
both host I/O precisions; configuration checks exercise the production validation after the
configuration boundary double. Real parser/SDK integration, the production compiler, GPU parity
and latency still require supported-system tests. No inference scheduling or performance feature
was added. The patch must be rebuilt and reinstalled before testing the updated executable.

## Remove unused I/O paths (2026-10-20)

Removed `explicit_tt`, `use_fp16` and unused `dataDirs` from the app. Only one FP32
image input and one FP32 modal output are accepted; internal TensorRT precision is
unchanged. Preprocessing, synchronous execution, handoff and controls are unchanged.
Existing configs must remove all three keys before startup; the exporter/sample no
longer emit them. No installed config or binary is changed by this source update.

Local normal, optimized and ASan/UBSan suites pass 26 cases / 447 assertions,
including startup/reload rejection of auxiliary and FP16 tensors with cleanup.
Three exporter config/parity CPU checks pass; the ONNX-dependent export test could
not run because ONNX is unavailable locally. Production GPU build/parity remain
operator checks. This cleanup is on the correctness branch, separate from pinned input.

## Startup engine discovery and notebook catalog export (2026-10-20)

The user chose startup-only discovery. Reuse `validateEngine` and the existing INDI
selection helper; no rescan control or activation change. `.trt`, `.engine`, `.plan`
files are validated sequentially against the startup engine; invalid candidates are
logged and omitted. Existing explicit model entries remain extra candidates.
The notebook helper preserves its actual internal-FP16/FP32-I/O build flags and
publishes uniquely named complete files atomically to the shared catalog. The helper
sets directory read/execute and engine-read ACLs for xsup. No installed config changes,
GPU builds, app restarts, or inference were performed by the agent.

Local validation: 27 app cases / 461 assertions pass normally, with `-O3 -ffast-math`,
and with ASan/UBSan. Two standard-library-only exporter tests cover success, missing
SavedModel, failed conversion/build/ACL setup, and empty output; no GPU compiler was
run. Notebook update targets only the existing TensorRT export cell in the user's
`Train_bright.ipynb`, with a backup and original-file SHA check before replacement.

# Phase 8 offline architecture evidence

Validation completed 2026-11-06 on personal workstation `ASTR-LT-428.local`, macOS
26.3.2 arm64. Starting clean commit: `6ae20f859ce308d1d5734154a7c3e5c280a7efb4`
(committed Phase 7). Apple LLVM 21.0.0 (`clang-2100.0.123.102`), Homebrew GCC
15.2.0, Eigen pkg-config 5.0.1/header macros 3.5.0, mxlib 0.2.0. All new code is
experimental/offline and strict-math; no instrument, native app, service or DM
stream was accessed. Legacy and production execution remain unchanged.

Raw evidence: [synchronous reproduction](phase8-synchronous.csv) (92 workloads),
[Clang worker benchmark](phase8-async-clang.csv) and [GCC worker benchmark](phase8-async-gcc.csv)
(each 32 model rows / 16 scenarios).
The original component configurations, RNGs, 512 warmup and 2,000 timed records
are unchanged. Reproduced synchronous pair medians are 39.666/48.750 us at two
modes and 504.000/572.042 us at ten modes (RLS/QR); ten-mode pairs miss 1,391/2,000
and 2,000/2,000 frame budgets. Source math is not made faster by the scheduling
prototype: work is moved onto exclusive model owners, and bounded publication,
queueing and rejection are measured separately.

No CPU affinity, real-time priority, memory locking, frequency or background-load
control was used. Timed captures follow builds, rather than deliberately running
compilation alongside them. macOS measurements do not qualify Linux execution,
physical timing, native linking or a worst-case deadline. These are short laptop
observations; page faults, interrupts, compiler/Eigen changes and production
Legacy/INDI/transport work still need target measurements.

## Clang paced results

The tables select 8-kHz sensor scenarios; the raw CSV also retains all 4-kHz rows.
Both model workers run concurrently. Queries acquire and validate a complete
snapshot, multiply the full reference gain's nearest-action rows by a current
causal state, and release their pin. Values are microseconds. Availability is
valid queries out of 2,000; the maximum age limit is 2,000 us.

| Modes / fit / model, normal | Median | p95 | p99 | p99.9 | Maximum | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / rls / standard | 0.125 | 0.25 | 0.292 | 0.334 | 0.5 | 2000/2000 |
| 2 / rls / assisted | 0.166 | 0.208 | 0.291 | 0.417 | 1.291 | 2000/2000 |
| 2 / qr / standard | 0.125 | 0.292 | 0.333 | 0.5 | 1.791 | 2000/2000 |
| 2 / qr / assisted | 0.166 | 0.208 | 0.333 | 0.583 | 1.542 | 2000/2000 |
| 10 / rls / standard | 0.334 | 0.5 | 0.625 | 1.375 | 1.791 | 2000/2000 |
| 10 / rls / assisted | 0.292 | 0.417 | 0.5 | 0.834 | 3.917 | 1999/2000 |
| 10 / qr / standard | 0.375 | 0.5 | 0.625 | 2.25 | 32.042 | 2000/2000 |
| 10 / qr / assisted | 0.333 | 0.417 | 0.584 | 1.834 | 2.042 | 1998/2000 |

Training includes the owning manager's history/identification/rank/reference and
forecast diagnostics, with only the selected independent model adapting. Each
private manager's inactive half copies histories but never fits or checks rank.
The extra fixed storage/work is deliberately retained to reuse the committed
implementation; it is included in these observations.

| Modes / fit / model, normal | Training median / p99 / max | Publication p99 / max | Record-to-publication p99 | Queue high | Evidence age p99 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 / rls / standard | 7 / 11.209 / 41.208 | 0.542 / 1.042 | 291.042 | 1 | 751 / 753.917 |
| 2 / rls / assisted | 36.875 / 44.625 / 82.917 | 0.375 / 0.709 | 321.958 | 1 | 751.125 / 754.083 |
| 2 / qr / standard | 8.541 / 13.25 / 62.166 | 0.5 / 1 | 296.667 | 1 | 751.083 / 766.292 |
| 2 / qr / assisted | 44.958 / 56.584 / 140.667 | 0.417 / 11.5 | 333.375 | 1 | 751.208 / 768.583 |
| 10 / rls / standard | 170.458 / 189.833 / 246.333 | 1.291 / 1.459 | 458.916 | 1 | 751.25 / 755.875 |
| 10 / rls / assisted | 337.958 / 363.125 / 395.417 | 1.708 / 1.917 | 631.416 | 2 | 1085.708 / 1136.75 |
| 10 / qr / standard | 198.125 / 276.583 / 426.916 | 1.417 / 1.959 | 498.5 | 6 | 751.541 / 1130.125 |
| 10 / qr / assisted | 380.375 / 473.917 / 687.459 | 1.709 / 7.25 | 689.5 | 6 | 1139.708 / 1248.75 |

| Modes / fit / model, stress | Query p99 / max | Valid | Stale | Common queue overflows | Discarded lane records | Observed evidence age max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 / rls / standard | 0.333 / 0.667 | 1814/2000 | 5 | 25 | 8 | 17169.625 |
| 2 / rls / assisted | 0.25 / 1.417 | 1452/2000 | 5 | 25 | 8 | 17169.708 |
| 2 / qr / standard | 0.292 / 0.709 | 1814/2000 | 5 | 25 | 8 | 17245.5 |
| 2 / qr / assisted | 0.25 / 1.541 | 1452/2000 | 5 | 25 | 8 | 17245.583 |
| 10 / rls / standard | 0.542 / 2 | 1335/2000 | 5 | 24 | 8 | 16671 |
| 10 / rls / assisted | 0.459 / 1.708 | 1027/2000 | 5 | 24 | 8 | 16671.084 |
| 10 / qr / standard | 0.667 / 1.875 | 1334/2000 | 5 | 25 | 8 | 17027.25 |
| 10 / qr / assisted | 0.458 / 2.917 | 1026/2000 | 5 | 25 | 8 | 17027.458 |

Full frontend execution includes causal state construction, raw sample generation,
both bounded fanouts/queries and numeric comparison. Across the 16 Clang scenarios,
its maximum is 49.333 us, with zero measured execution or absolute scheduled
500-us deadline misses. This does not imply a target worst-case bound. Training
and publication can finish late without blocking a frontend query. Normal
readiness rejections at ten modes are retained, rather than weakening rank checks.

Stress injects independent 8/16-ms worker stalls, 90-us producer scheduling jitter,
capacity eight (normal 32), missing sensor/common timing, sequence gaps, two
independent resets and a source restart. All runs use 512 warmup plus 2,000 timed
frames. Each profile generates 5,024 (4 kHz) or 10,048 (8 kHz) individually
timestamped sensor events. Sensor generation is batched per WFS interval in the
offline producer; it is not a separately scheduled hardware acquisition thread.
The latest sample preceding the exposure end is used, with a declared 1-us
synthetic sensor availability delay. No filtering/calibration or physical DM
application is established by this generator.

Queues include in-flight entries. A full queue rejects the newest record from
both lanes, invalidates both generations immediately and lets consumers discard
old prefixes safely. A fast standard worker therefore also loses availability
when assistance saturates; this explicit common-identity policy is a limitation,
not a blocking wait. Prefixes processed before the barrier can differ between
workers. Their learned weights are retained on history breaks, while readiness
must reacquire full uninterrupted evidence. Full resets clear weights on their
owner and reject all older publications/queued prefixes. Overflow/restart fitting
traces need not match a hypothetical no-drop synchronous fit.

The `age` cohort includes every observed snapshot's metadata, even if its
prediction is rejected. Stress ages above 2,000 us are not served valid outputs.
`valid` counts successful predictions; stale/busy/unready/invalidated columns do
not exhaust all rejection reasons (e.g. unavailable state/no snapshot/time order).
Input/queue counters include warmup; training quantiles use post-warmup processed
records and include reacquisition. Trace buffers are fixed; dropped telemetry and
unprocessed/discarded records are reported. No pending work or trace loss remains
in the captured Clang runs after the bounded coordinator drain.

## GCC portability result

The same strict GNU 15.2.0 Mac build keeps the frontend bounded in the measured
run (maximum 80 us; zero observed execution/absolute deadline misses and allocation
requests), but **does not sustain ten-mode training**. The normal standard-worker
p99 is 1,196–1,265 us across learner/rate profiles, exceeding the 500-us period.
Normal ten-mode runs have 16–18 common queue overflows (including warmup), only
11–12/2,000 valid standard predictions, and **0/2,000 assisted predictions**.

Those overflows repeatedly break the assisted model's evidence window before
its larger rank gate can become ready. Its low training quantiles in the raw
CSV therefore measure mostly warming/identification, not a ready assisted rank
workload. They must not be presented as a speed improvement. Extra queue capacity
would delay rejection without curing sustained throughput. This remains an
explicit compiler/platform qualification blocker; both workers are required to
keep up with the admitted record rate for reliable availability. Per-lane
independent drop policies would change the common-history contract and require
separate approval/validation. No rank tolerance, arithmetic, production flag or
adaptation cadence is relaxed to hide the result. Exact committed numerical
regressions pass under this compiler despite the performance limit.

A Clang ASan/UBSan paced run also covers all 16 scenarios (100 timed frames each),
with zero observed allocation requests and no sanitizer findings; its timing is
not a performance qualification. ThreadSanitizer separately checks actual pinned
snapshot reuse and reset races against committed per-version outputs.

## Numerical behavior and allocation scope

The actual committed Phase 7 code is extracted from local Git objects and built
under separate symbols. Exact ordered comparisons span 4,800 records, modes=2/10,
RLS/QR, gamma=1/.99, missing data, gaps and reset. Canonical coefficients, origin
proposals, frozen-origin RMSE, rank/readiness/status and fit versions match with
zero tolerance. The actual threaded test compares each observed coherent snapshot
to its committed version, including repeated, backlogged, independent resets.
The controlled delayed-publication test evaluates an old gain on a current state:
matched-version error is zero; maximum difference from the current-fit proposal
is 0.0352369. Normal benchmark current-fit differences reflect publication lag;
stress differences also include rejected histories and reset timing. They are
not numerical regression tolerances or predictions of physical performance.

All 32 Clang rows report zero observed frontend Eigen/C++ allocation requests and
zero worker requests at gamma=1. Slot/queue/trace/state buffers are preallocated;
startup allocations are outside timing. Gamma=.99 conventional RLS retains one
worker-side Eigen temporary per adaptation for exact Phase 7 rounding, as before.
Readers never allocate, wait on a worker or spin/retry a failed pin. The allocation
probe observes Eigen permission checks and ordinary C++ new/new[]; it excludes
arbitrary libc/aligned allocation, OS/runtime activity, peak heap and Eigen byte
sizes. The known allocation self-test passes before capture. The shared Eigen
no-malloc switch is not toggled concurrently. Lifecycle join/drain and formatting
are outside frame timing. Worker control/polling costs are represented in
end-to-end delay, not the isolated manager.process timer.

## Final validation results

| Check | Result |
| --- | --- |
| Reproduced committed Phase 7 default suite | 333,616 assertions / 50 cases pass. |
| Final strict Clang | 823,820 assertions / 55 cases pass. |
| Final strict GCC | 822,490 / 55 pass. |
| Final Clang ASan + UBSan | 826,215 / 55 pass, no findings. |
| ThreadSanitizer concurrent reuse/reset case | 390,496 assertions / 1 case pass across both modes and learners, no reported races. |
| Clang Legacy `-O3 -ffast-math` | 186,615 / 9 pass. |
| GCC Legacy `-O3 -ffast-math` | Existing failure preserved: 8/9 cases pass; 182,253/182,254 assertions pass. Historical QR nonfinite fit at t=31; original/clone outputs still match. |
| Native-helper mocked orchestration | Seven Python tests pass in each full `check`; not Linux build evidence. |
| Original source preservation | All 15 historical `loPredCtrl` files still match pinned tree; production math/defaults/app loop unchanged. |
| Original allocation contract after shared-probe extraction | All eight Phase 7 profiles pass, including cold/reacquisition, pause/reset, cached success/rejection. |
| Paced benchmark | Clang/GCC each 16 scenarios x 2,000 measured frames; ASan/UBSan 16 x 100; no observed frontend/worker requests at gamma=1, no sanitizer findings. |

Threaded assertion totals depend on how many publications are observed; they are
not fixed numerical tolerances. All exact coefficient/proposal/RMSE assertions
remain. Logs live under ignored `apps/loPredCtrlAcc/build/phase8-*`; captures above
are versioned separately. Previous initialized-VLA demonstration restrictions,
external dependency warnings, native linking and GNU default fast-math limitations
remain outside this prototype. Existing fast failure is neither waived nor made
passing with a flag change.

## Reproduction and Linux qualification

No native Linux runtime/VM/container is available here (Docker, Podman, Colima,
Lima, QEMU and Multipass were not found). Native app compilation and Linux worker
execution remain unverified. GCC results on this Mac are not Linux evidence.
On an approved isolated non-instrument Linux workstation with real Eigen/mxlib
and normal repository dependencies, run from the repository root:

```sh
hostname  # Stop if the host begins with exao; never use production instrument hosts.
uname -s
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/offline-linux-gcc CXX=g++ check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/offline-linux-clang CXX=clang++ check
make -C apps/loPredCtrlAcc -f tests/Makefile.async -j2 \
  BUILD_DIR=build/async-linux-gcc CXX=g++
apps/loPredCtrlAcc/build/async-linux-gcc/ddspc_async_benchmark 2000 \
  > apps/loPredCtrlAcc/build/async-linux-gcc.csv
make -C apps/loPredCtrlAcc -f tests/Makefile.async -j2 \
  BUILD_DIR=build/async-linux-clang CXX=clang++
apps/loPredCtrlAcc/build/async-linux-clang/ddspc_async_benchmark 2000 \
  > apps/loPredCtrlAcc/build/async-linux-clang.csv
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/offline-linux-asan CXX=clang++ \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/offline-linux-tsan CXX=clang++ \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=thread -fno-omit-frame-pointer'
apps/loPredCtrlAcc/build/offline-linux-tsan/ddspc_offline_test '[async][threaded]'
python3 apps/loPredCtrlAcc/tests/native_linux_build.py \
  --build-root /tmp/ddspc-native-phase8 --cxx g++ --jobs 2
```

Use fresh build directories when changing compiler/flags. The native helper's
output directory must not exist. It builds/links the real profiles without
installing or executing them. Preserve the documented GNU default fast-math QR
nonfinite failure; strict worker success must not hide it or change production
flags. Repeat queue/age/overflow tests and measure under sustained Linux CPU/I/O
load, with actual scheduling/clock/page-fault/affinity constraints documented.
Any concurrent library/runtime changes or a new snapshot reader require a new
ownership/race review; this protocol supports exactly one frontend reader.

The single frontend is an explicit ownership restriction: real WFS, applied-input,
sensor and INDI callbacks may be multiple producers and need a reviewed bounded
aggregation/control design. Configuration is immutable; replace/quiesce the
prototype outside prediction to change dimensions or mathematics.

Before production integration, resolve applied-input/exposure/availability
provenance, raw sensor clock mapping, model-age policy, common-overflow versus
per-lane availability policy, calibrated sensor reduction, target CPU/memory and
sustained worker throughput. The source has no production asynchronous entrypoint
or new actuation route. Further integration requires explicit user approval.

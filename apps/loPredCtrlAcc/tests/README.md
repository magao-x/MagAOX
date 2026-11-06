# Offline DDSPC validation

Run on a personal workstation or isolated development environment. Nothing here
requires instrument access, running the native application, or mounting DM streams.
The numerical executables link Eigen and the existing mxlib headers; application
tests use stream-free lifecycle doubles and real INDI property objects.

From the repository root:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-legacy-fast
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 \
  BUILD_DIR=build/offline-sanitize \
  CXXFLAGS='-std=c++20 -O1 -g -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check
apps/loPredCtrlAcc/build/offline/ddspc_offline_test '[.legacy-drift]'
apps/loPredCtrlAcc/build/offline/ddspc_offline_test '[.legacy-tiny-rotation]'
```

`prepare_legacy.py` verifies all 15 original `loPredCtrl` source files against the
pinned historical Git tree before building its independent replay. No branch
checkout, network fetch, commit or instrument command occurs. Preserve the
default bit-exact assertions. The optional drift diagnostic **does not accept**
clone drift: it measures it separately while requiring the original/restored
pair to agree. Phase 5's two strict GCC QR failures are corrected by dispatching
to separate compile-time Legacy/causal Givens kernels. The exact tests remain
unchanged; direct factor/gain/weight and single-rotation comparisons were added.
On GCC 15.2 AArch64, an inner runtime structural-zero branch changed the complex
loop vectorizer's operand/contraction order even when false. Keeping that branch
out of the Legacy loop restores the historical arithmetic. `-ffp-contract=off`
alone did not fix the old kernel; disabling loop vectorization did. No global
floating-point flags, Legacy indexing, cost, feedback or defaults were changed.

The GNU `-O3 -ffast-math` application replay still exposes an existing historical
QR failure: tiny rounding remnants cause reciprocal overflow in Givens rotations
and nonfinite learned coefficients. An exponent-bit oracle now detects this even
when `allFinite()` is optimized away by finite-only assumptions. Exact comparison
assertions remain. `-fno-reciprocal-math` makes the GNU fast suite pass in a separate
experiment; it is **not** added to production defaults. Original `loPredCtrl`
is unchanged. This issue and the native Linux build remain commissioning blockers.
Use a new `BUILD_DIR` when changing compilers or flags.

The hidden tiny-rotation case passes with the tested strict compilers and Clang
fast build, and deliberately fails the finite assertion with GNU default fast
flags. The test-only GNU workaround can be reproduced with:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j3 \
  BUILD_DIR=build/offline-gcc-norecip CXX=g++-15 \
  LEGACY_FAST_FLAGS='-O3 -g -std=c++20 -Wall -Wextra -ffast-math -fno-reciprocal-math' check-legacy-fast
```

`g++-15` names this workstation's GNU compiler; use the approved compiler on the
development host. This command neither changes native build defaults nor runs
the native application.

`check` also runs seven Python build-tool tests. Their compiler/Git/make steps
are mocked; they verify orchestration and never establish native compilation.
Only the generated metadata-copy wrapper is actually executed in those tests.
Run them separately with:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-build-tool
```

`comparison.cpp` uses the accelerometer-disabled clone's Legacy path (covered by
the pinned-source Clang replay) and generates deterministic two-mode plant trajectories with a
one-step synthetic held-input contract, 37/83/211-Hz components and independent
broadband innovations. It fits Legacy, corrected standard, and assisted models on
the same 1,800 records and mature update counts, then freezes all fits. Six
hundred held-out samples score forecasts conditioned on **actual future inputs**;
those future inputs are never decision features. Separate frozen-fit feedback
trajectories share the same exogenous realization, with clipping and periodic
holds. Their resulting commands and residuals naturally differ. Absolute modal
commands, zero added legacy feedback, and disabled app output accumulation are
explicit simulation conventions; these are not an on-sky configuration or an
optical/mechanical simulation. Sensor zero-coupling and unrelated-sensor negative
controls prevent a claim of universal accelerometer benefit. F=3 includes the
unmodified Legacy multichannel indexing defect and unpredictable future
disturbance innovations. The original app's scaled penalty is retained; corrected
models use literal lambda=0.3. Different costs are not tuned to equal command RMS.

Build and capture the strict-math component benchmark separately:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.benchmark -j3
apps/loPredCtrlAcc/build/benchmark/ddspc_benchmark 2000 \
  > apps/loPredCtrlAcc/build/latency.csv \
  2> apps/loPredCtrlAcc/build/latency-environment.log
```

The benchmark covers two and ten modes, N=5, F=3, two sensor channels and Ha=20,
both RLS methods, isolated core work, the managed pair, and synchronized builder
with/without both models. It warms 512 records and times 2,000 more by default.
Phase 6 adds isolated history, reference-law and normalized pivoted-QR rank
workloads, bringing the CSV to 48 rows. Rank-only cases use frozen independent
seeded matrices; they do not change the original full-workload input sequences.
They identify component costs without pretending to measure all pivot/conditioning
patterns. Every measured pair must reach readiness. CSV reports p50/p95/p99/p99.9/max and
counts records exceeding a **whole-frame** 500-us budget. Timings include
benchmark input assignment/generation where present and the allocation probe;
builder workloads use pre-generated immutable records. They exclude native
transports, exposure/actuation latency, app/INDI locks, scheduling and calibration.
No target-machine worst-case real-time guarantee follows from laptop timings.

`allocation_probe.hpp` observes Eigen's allocation permission checks, including
its direct-malloc path, and separately counts ordinary C++ new/new[] requests.
A known Eigen allocation and a C++ allocation must each count once before any
report is emitted. Eigen permission checks are allocation **requests**, not a
system allocator trace (zero-size requests can differ by Eigen version).
Reported C++ bytes exclude Eigen allocation sizes; C malloc outside Eigen,
aligned C++ allocation, peak live heap, and OS/library allocation are not measured.
This proves hot-path allocation and is not an allocation-free certification.
The probe is force-included only in the separate benchmark build.

For a real Linux compile/link check, first provision the normal MagAO-X build
dependencies on an approved non-instrument development host. The existing
`.github/workflows/build-coverage.yml` identifies a development container image; no
container runtime is available on the present Mac. Then run:

```sh
python3 apps/loPredCtrlAcc/tests/native_linux_build.py \
  --build-root /tmp/ddspc-native-phase6 --cxx g++ --jobs 2
```

The new output directory must not exist. The script rejects macOS and `exao*`
hosts and output directories inside the source tree. It freezes versioned/nonignored
working source bytes once and copies that immutable snapshot into both profile trees.
It builds the real libraries, generated logger headers, PCH and app binary in
each. Legacy explicitly uses the normal `-ffast-math` default; shadow rebuilds everything
with `OPTIMIZE=-fno-fast-math`. The real Git-header generator captures original
source metadata once with optional index writes disabled; the profile wrappers
only copy that frozen header to confined staged paths. Both profiles therefore
use the same source and metadata even if the working tree changes during a build.
Compiler details, source/header hashes, command logs and linked binary hashes
remain in the output directory. Failure is recorded per profile and does not
skip the other attempt; a successful make without a binary is a failure.
Ignored local overrides and SDKs are not provisioned; source symlinks and Git
submodules require an explicit staging rule and are rejected. Inspect logs and
provide real dependencies instead of replacing real headers with test doubles. The script
does not install or execute the app, contact a host or change services. Its
Linux build path is **not yet validated**; seven offline orchestration/guard tests
pass, including source edits between profiles, injected failure, missing binaries,
and metadata confinement. Native Linux compatibility remains a blocker until both
real builds complete and their logs are reviewed.

The historical `testRLS` demos also run with `check-demos DEMO_NAMES=testRLS`.
The original/current `testPredCtrl` demos still have initialized variable-length
arrays rejected by this Clang C++20 compiler; they are not the application build
or the deterministic comparison oracle. Neither original source nor operational
controller code was changed to address that demonstration-only issue.

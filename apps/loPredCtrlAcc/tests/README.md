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
```

`prepare_legacy.py` verifies all 15 original `loPredCtrl` source files against the
pinned historical Git tree before building its independent replay. No branch
checkout, network fetch, commit or instrument command occurs. Preserve the
default bit-exact assertions. The optional drift diagnostic **does not accept**
clone drift: it measures it separately while requiring the original/restored
pair to agree. GCC 15 on Darwin fails two exact QR replay cases; that failure
remains visible, including with `-ffp-contract=off`. Classic RLS remains exact.
Use a new `BUILD_DIR` when changing compilers or flags.

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
Every measured pair must reach readiness. CSV reports p50/p95/p99/p99.9/max and
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
  --build-root /tmp/ddspc-native-phase5 --cxx g++ --jobs 2
```

The new output directory must not exist. The script rejects macOS and `exao*`
hosts, copies versioned/nonignored working source files into two new snapshots,
and builds the real libraries, generated logger headers, PCH and app binary in
each. Legacy explicitly uses the normal `-ffast-math` optimization default; shadow rebuilds everything
with `OPTIMIZE=-fno-fast-math`. The real Git-header generator reads the original
source metadata with optional index writes disabled; it may write only the
staged output header. Compiler details, source hashes, command logs and linked
binary hashes remain in the output directory. Ignored local build overrides,
SDKs and Git submodule contents are not provisioned; inspect logs and provide
dependencies instead of replacing real headers with test doubles. The script
does not install or execute the app, contact a host or change services. Its
Linux build path is **not yet validated**; the macOS refusal path and Python
syntax were checked. Native Linux compatibility remains a blocker until both
real builds complete and their logs are reviewed.

The historical `testRLS` demos also run with `check-demos DEMO_NAMES=testRLS`.
The original/current `testPredCtrl` demos still have initialized variable-length
arrays rejected by this Clang C++20 compiler; they are not the application build
or the deterministic comparison oracle. Neither original source nor operational
controller code was changed to address that demonstration-only issue.

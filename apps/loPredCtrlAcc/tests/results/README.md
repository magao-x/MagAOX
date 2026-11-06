# Phase 7 offline benchmark evidence

See [Phase 8](phase8.md) for the subsequent independent-worker prototype,
reproduced synchronous baseline, paced publication/queue benchmarks and Linux
qualification commands. The Phase 7 captures below remain unchanged.

[Before](phase7-before.csv): unchanged Phase 6 source at
`352446f83737573023870096d2a5388859afcd3e`, 48 workloads.
[After](phase7-after.csv): Phase 7 working source, 92 workloads including all
original 48 with their original dimensions, flags, seeds and record sequences.
The benchmark executable compiled before editing was retained for the baseline.
Both captures use 512 warmup and 2,000 measured iterations per workload.

Environment: personal workstation `ASTR-LT-428.local`, macOS 26.3.2 arm64,
Apple LLVM 21.0.0 (`clang-2100.0.123.102`), Eigen pkg-config 5.0.1 (header macros
3.5.0), mxlib 0.2.0. Benchmark flags: `-O3 -g -std=c++20 -Wall -Wextra
-fno-fast-math`, `EIGEN_DONT_PARALLELIZE` and the force-included allocation probe.
N=5, F=3, modes=2/10, two accelerometer channels, Ha=20, gamma=1, prior=0.1,
literal lambda=0.3, both classical and QR RLS. The final capture follows builds;
no build was deliberately launched alongside the measured after run. No CPU
affinity, frequency, thermal or background-load controls were applied. The before
capture has substantial scheduler tails. Tail improvement cannot be attributed
solely to the optimization, and no target-machine real-time bound is established.

CSV latencies are microseconds; quantiles use the original sorted-sample index
rule. `over_500us` counts violations of the entire WFS frame period. Every measured
pair reaches both-model readiness. The original value-returning core/reference
APIs remain measured under their old names; the additional `fixed_*`,
`*_fixed_only` rows explicitly opt into preallocated storage. For isolated
core comparisons, compare each old `*_step_identify_reference` baseline to the
new `*_fixed_step_identify_reference`. Both use identical seeded records.
`*_rank_only` deliberately retains the old allocation-heavy isolated operation;
compare its baseline to `*_rank_fixed_only`. The managed-pair and builder-plus-pair
rows directly measure the optimized manager under their original names.

`*_cached_predict` measures only a ready, latest-frame matrix-vector prediction
with a finite preallocated canonical state of ones and output buffer. It excludes
state construction, learner updates, snapshot handoff and transport; the sub-us
values are affected by timer resolution. It is not an accuracy benchmark or a
concurrent worker implementation. `*_managed_pair_frozen` disables adaptation
while retaining every-record rank/readiness and frozen-origin diagnostics; rank
continues to dominate.

`eigen_requests_per_record` observes Eigen allocation-permission checks, not
allocated bytes or a system allocator trace. `cpp_new_per_record` and
`cpp_bytes_per_record` count ordinary C++ new/new[] only, excluding Eigen malloc,
aligned allocation, unrelated libc allocation and peak live memory. A known Eigen
and C++ allocation self-test must pass before reporting. Gamma=0.99 is covered by
the separate no-malloc/allocation contract, not these original gamma=1 timings:
classical forgetting RLS retains one Eigen temporary per adapting learner update
to preserve exact expression rounding. QR, cached prediction, paused fitting and
explicit reset report zero requests at both tested forgetting factors.

The unchanged builder still requests six Eigen allocations plus one C++ allocation
(312 observed bytes) per result, so the combined acquisition/assembly path is not
allocation-free. Exception construction and arbitrary library/OS fault handling
are not bounded by these numerical fixtures. See the [tests README](../README.md)
and [engineering plan](../../../../agents/plans/2026-11/accel-ddspc-controller.md)
for numerical/sanitizer validation and the remaining Linux/physical-timing gates.

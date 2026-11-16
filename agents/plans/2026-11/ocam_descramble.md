# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
<!-- A brief statement of the problem -->

The ocam2KCtrl (which is for camwfs) has a costly "descramble" step.  We want to investigate whether that can be sped up.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

The total processing time for camwfs is ~37 us, with 1.5 us of jitter.  We have recently put some work into optimizing memory management for cache access.  Previous measurements on older hardware showed that descramble was 10 us slower than a pure memcpy.

The most likely thing to test is to have a second thread on a neighbor core dedicated to  half the descramble.  See rtSetup/RTC/make_cpusets for the current cpu configuration. The cpu is a AMD Ryzen Threadripper PRO 5995WX 64-Cores

But there may be other optimizations upon a deep look at the code.  The SDK is vendored under apps/ocam2KCtrl/fli.


## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->
The main requirement is to not be slower.  Whether or not any given allocation of systems resources is worth it depends on the speedup.

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->
It would be good to have a minimally invasive way to A/B test, e.g. a flag we can set in INDI to go back and forth.

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->
This is focused on the descramble step, but other optimizations are uncovered in the investigation that are applicable to other cameras we should apply them to libMagAOX.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Investigation date: 2026-11-16. Source baseline: `dev`, `cbc1d5cf`.
Host verified as `mx11`. This pass inspects local source and configuration
only; no instrument commands, benchmarks, or implementation were run.
Status: **single-core implementation complete; first RTC comparison shows lower mean and higher jitter, so qualification remains open**.
The answers below and the subsequent user request authorize phases 1–3.
The second-thread experiment remains deferred.

## Task Summary

Reduce camwfs's acquisition-to-publication cost without degrading image
correctness, jitter, frame delivery, or end-to-end AO latency. Retain the SDK
descrambler as the default and provide a live INDI comparison with qualified
alternatives. Start with measured single-core candidates, then evaluate a
persistent second worker if its gain justifies dedicating another core.

### Findings from the current code

1. **The hot path is a gather with sequential output.**
   [ocam2KCtrl::loadImageIntoStream](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp)
   calls the SDK, optionally followed by digital binning. In
   [ocam2_descramble_binning2x2](../../../apps/ocam2KCtrl/fli/ocam2_sdk.c),
   7,200 iterations produce two 7,200-pixel output chunks:
   `out[i] = raw[table[2*i]]` and
   `out[7200+i] = raw[table[43200+2*i]]`.
   The 120x120 output is 28,800 bytes. The SDK's full normal-mode `int`
   table is 230,400 bytes; bin2 accesses alternating entries in two parts
   spanning 115,200 bytes of cache lines with an aligned table base.
   The old memcpy comment is historical evidence, not a
   measurement on the present RTC.

2. **The local table is structured, so a second core is not the only option.**
   Read-only inspection of
   `/opt/MagAOX/config/ocam2k/fli/ocam2_descrambling.txt` found 57,600 unique
   indices, range 576–63,887. The bin2 selection has 14,400 indices, range
   577–32,207. Each output row comprises four 30-pixel affine runs with
   strides of +16 or -16 raw samples. These support two candidates:
   a packed per-mode index table, and a validated structured kernel that
   removes per-pixel table loads. SIMD/deinterleave variants are experiments,
   not assumed improvements. Validate any structural assumption against
   every selected index at configuration time.
   Local table SHA-256:
   `477a1aa37cab3d9cd9b087cce3204675c6e45cbea904ec1a20e754a48602b324`.

3. **Naively splitting the output duplicates the source working set.**
   With the local bin2 table, each top/bottom output half touches the same
   930 raw cache lines (64-byte lines, assuming aligned raw storage).
   Splitting the existing SDK loop in half has the same issue. Output rows
   `y` and `119-y` draw from the same raw row; partitioning these paired
   rows can separate source accesses. A 30/30 raw-row split introduces
   output boundaries halfway through cache lines even with an aligned
   output base. Benchmark that layout against an aligned 28/32 split
   and a simple output split, accounting for the actual buffer addresses.
   Sharing read-only input is correct, but its cache cost must be measured.

4. **Staging has already been generalized.**
   [frameGrabber](../../../libMagAOX/app/dev/frameGrabber.hpp) sets
   `md->write = 1` before calling `loadImageIntoStream`, then uses the
   existing private buffer and `MagAOX::streamCopy` when staging is enabled.
   The local `camwfs.conf` selects `bin2`, `framegrabber.staged=true`,
   64 EDT buffers, FIFO priority 90, and cpuset `camwfs`.
   `stagedCopyNT` defaults to true. Keep this path and its ordering;
   do not add another staging copy. The
   [previous memory investigation](shmim-monitor-speedups.md) records the
   approximately 37 us result with this arrangement. Local configuration
   is evidence of intended setup, not confirmation of live RTC state.

5. **No unused core currently shares camwfs's L3 cache.**
   [make_cpusets](../../../rtSetup/RTC/make_cpusets) assigns CPUs 32–39 to
   `camwfs_sw`, `camwfs`, `camwfs_irq`, `aol1_modevalWFS-sw`,
   `dmtweeter-recon`, `dm00comb`, `woofer`, and `dm01comb`, respectively.
   Camera acquisition is on 33; EDT IRQs belong on 34. Spares 26–31 and
   46 are in different L3 domains. A neighbor worker needs an explicit
   reassignment; do not put it on the IRQ core or silently share another
   real-time task's core.

6. **Existing timing can qualify the complete path.**
   `fg_timing.delta_aw` measures SDK acquisition timestamp to stream write
   timestamp, not pure descramble time. `fg_stage_timing.conversion`
   brackets validation completion through loading and the staging copy;
   it also includes digital binning when active. Current-frame semaphore
   posts and deferred EDT rearm follow the write timestamp. Use these
   existing metrics with end-to-end loop timing; isolate the kernel in
   an offline benchmark first. The local config disables stage timing,
   so comparisons must explicitly record whether it is enabled.

7. **Current app tests do not verify the real mapping.**
   [ocam2KCtrl_test.cpp](../../../apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp)
   stubs `ocam2_descramble` as a copy. A separate suite linked to the real
   vendored SDK is necessary. The app selects SDK normal, 2x2, 1x3 and
   1x4 modes from raw height; its `bin3` and `bin4` configurations then
   apply digital X binning. They do not use the SDK's 3x3/4x4 functions.

## Key Assumptions

- Bin2 is the first performance target. Other modes retain the SDK path
  until separately qualified. Raw frame sizes and bit patterns, output
  orientation, frame counters, timestamps and publication order are unchanged.
- The configured SDK table remains the source of truth. Do not hardcode
  the observed table or infer its validity from a filename.
- An offline benchmark can screen candidates, but cannot reproduce EDT DMA,
  RTC scheduling, or reader-induced cache traffic. Only operator-run RTC
  comparisons can establish the production improvement.
- Retain staged publication and its copy settings during comparisons. No
  claims of faster processing should be made from the 37 us figure alone:
  first establish which interval and jitter statistic are being compared.
- Keep the vendor implementation unchanged as an independent reference.
  A small app-local helper is preferable to a new base class for a single
  camera. Promote a utility to libMagAOX only if the investigation identifies
  a concrete shared use; staging and non-temporal copy already live there.

## Requirements

1. Byte-for-byte equivalence to the current SDK for each enabled fast path,
   including all 16-bit pixel patterns. Preserve current digital-bin behavior.
2. Default to the SDK. Unsupported modes or failed candidate preparation
   leave the SDK available and publish a clear unavailable/error status.
3. Change implementation between frames without camera serial commands,
   acquisition restart, stream recreation, or a mixed-implementation frame.
   Report requested and applied selections separately.
4. Allocate maps, scratch storage and worker state outside the frame loop.
   No per-frame thread creation, dynamic allocation, logging, or INDI writes.
5. Keep EDT acquisition/rearm, SDK lifecycle, and stream publication on their
   current owner. Finish all pixel work before copying/publishing or allowing
   source, destination, or mapping storage to be changed or reused.
6. Accept a candidate only after matched RTC comparisons show no repeatable
   regression in camera mean, jitter/tails, frame delivery, or loop latency.
   A kernel-only win is insufficient. Preserve a quick return to the SDK.

## Questions and Points of Clarification

1. **Review gate:** approve the staged approach below: first the real-SDK
   tests, offline benchmark and single-core candidates; then the worker
   experiment if the results justify it. No implementation begins before
   this review, as requested above.

   ANSWER: approved.

2. **Before a worker trial:** which CPU in 32–39 can be reassigned, and which
   current task should move? Recommendation: defer this decision until the
   single-core measurements are available; preserve CPU 34 for EDT IRQs.
   No CPU allocation decision is required for the first phase.

   ANSWER: agree.  For testing we don't need any other apps running, so we can just temporarily shut one down.

3. **Before RTC acceptance:** confirm the reported 37 us / 1.5 us metric,
   active camera mode/FPS, staging settings and reader workload. Agree on
   the minimum worthwhile gain for dedicating a core after seeing the
   benchmark results, rather than imposing an arbitrary threshold now.

   ANSWER: agree

## Tests

### Correctness and lifecycle

- Add a hardware-independent suite using the real SDK and deterministic
  table fixtures; optionally run it with the supplied configuration table.
  Compare full output buffers against the unchanged SDK for index ramps,
  impulses, alternating bit patterns, signed extrema and seeded random data.
  Check sentinels, source immutability, minimum raw sizes, and every split
  boundary. The SDK frame-number result must agree wherever that API is used;
  app frame-number validation remains unchanged.
- Exercise normal, 2x2, 1x3 and 1x4 mapping preparation and verify explicit SDK
  fallback for unqualified modes. Verify structural-kernel rejection on
  valid tables with a different layout. Use sanitizers for bounds/lifetime
  checks, independently of performance builds.
- Extend the existing app harness for default/invalid selections,
  requested-to-applied acknowledgment, repeated switches, configuration
  failure, recovery, shutdown and power-off. Use nonuniform pixels for
  digital binning: the existing constant-value case cannot detect changed
  sample selection.
- If the worker is pursued, deliberately delay it and verify no partial
  frame is published, no next job overwrites a pending job, and no buffer/map
  is freed or reused while it remains referenced. Cover startup/affinity
  failures, switching back to SDK, and shutdown/reconfiguration with an
  outstanding job. Fault handling must quiesce the worker before fallback
  can reuse the destination; a timeout alone does not make reuse safe.
- Reuse existing staging integration and streamCopy tests. Run the OCAM
  suites affected by lifecycle changes, frameGrabber integration tests,
  and a build against the available EDT SDK without launching the app.
  Register new suites in `tests/tests.list` with `application_unit_test`
  grouping and explicit real-API Doxygen links where indirection requires them.

### Performance and operator qualification

- Build a standalone, hardware-free benchmark accepting the actual table,
  mode, iteration count, and CPU choices. Compare the SDK, packed-map,
  structured scalar/SIMD candidates, and later the two-thread candidate.
  Include a same-size memcpy calibration, not as a valid image algorithm.
- Time pure conversion and conversion plus the existing staging copy.
  Two-thread results include dispatch and completion wait; also collect
  worker/main durations for diagnosis. Keep checksums/output comparisons
  outside timed intervals and prevent the compiler eliminating the work.
- Record compiler/options, CPU model/affinity, table hash, buffer sizes and
  alignment. Use production optimization, not coverage/debug. Preallocate
  and warm up; test a single hot buffer and a rotating 64-buffer input set
  matching the local EDT configuration. Rotation is a cache experiment,
  not a claim to simulate DMA.
- Report mean, median, standard deviation, p95/p99 and maximum, with sample
  counts and repeated interleaved comparisons. Measure timing overhead.
  Keep diagnostic instrumentation identical across candidates.
- The operator runs matched SDK/candidate/SDK windows on the RTC with the
  same FPS, mode, gain, reader workload, IRQ placement and copy configuration.
  Record applied mode, `delta_aw` and jitter, conversion if stage timing is
  enabled, frame gaps/failures and end-to-end loop timing. Compare with usual
  camwfs readers active, because they can change the writer's cost.
- After a confirmed mode switch, exclude at least one complete rolling
  timing window plus warm-up. Do not pool overlapping windows as independent
  samples or infer percentiles from mean/jitter summaries. Do not call
  `frameGrabberTiming::newSession()` during a load: it would reject the
  active attempt. Existing metrics need no schema change for the first trial.

## Implementation Plan

1. **Establish the reference and measurement harness.** After review, create
   `jrmales/ocam-descramble`. Add the real-SDK equivalence suite and standalone
   benchmark before changing application dispatch. Record baseline results
   here, including build options and table identity.

2. **Prepare a compact map and test single-core kernels.** Prefer deriving
   the per-mode map through the SDK itself: feed it a synthetic raw buffer
   whose 16-bit values encode their source indices, and retain the descrambled
   bit patterns as source indices. All four app raw frame sizes fit in
   16-bit indices (normal 63,888; bin2 32,736; 1x3 21,648; 1x4 16,368).
   Do this once during configuration with a valid SDK table, ignoring the
   synthetic frame-number output. This avoids a second CSV parser, duplicated
   mode formulas and reliance on the SDK's private globals. A bin2 map then
   occupies 28,800 bytes. Preserve bit patterns explicitly when crossing
   signed/unsigned storage, and check generated indices against raw size.
   This technique does not repair the SDK's existing malformed-table hazards.

   Benchmark the packed scalar gather first. Derive/validate affine runs
   from the map and compare a table-free scalar traversal; add SIMD or
   prefetch only where assembly and measurements justify it. Do not assume
   widening 16-bit loads into a vector gather is faster or bounds-safe.
   Retain only candidates that improve the complete measured path.

3. **Expose the first qualified candidate for live comparison.** Add an
   app-local selector, proposed config `camera.descrambleMethod` with
   default `sdk`, and INDI `descramble_method` with `sdk` / `optimized`
   choices. Provide applied-method and availability readback using existing
   INDI helpers. The callback only requests a change; the frame thread latches
   it once for the whole load, and ordinary app logic publishes readback.
   Requests cannot change tables or worker ownership in mid-frame. Preparation
   and selection failure leave the known working path available. Use an
   app-local header with out-of-class inline definitions, matching app/test
   conventions; leave vendored SDK files unchanged.

4. **Evaluate the persistent worker after the single-core results.** Reuse
   `MagAOXApp::threadStart` and the verified scheduling/affinity pattern in
   [bmcCtrl](../../../apps/bmcCtrl/bmcCtrl.hpp). `threadStart` can report
   success despite scheduling failure, so verify actual policy, priority
   and CPU placement before enabling a spinning worker. Use a startup
   barrier, preallocated mandatory-job state, and release/acquire job and
   completion counters on separate cache lines. A dropping/coalescing
   telemetry queue is not suitable for mandatory half-frame work.

   The helper reads immutable input/map data and writes only its assigned
   private-output regions. Benchmark the simple half split and paired-raw-row
   split described above. Include the cost of the final primary-thread copy
   reading cache lines owned by the helper. Account for the actual staging
   buffer alignment: `std::vector<char>` does not guarantee cache-line
   alignment. Keep spin checks responsive to stop/park requests; never spin
   on the acquisition core or a core with a conflicting FIFO task.

   When `sdk` is applied, park the worker so the baseline does not silently
   include an idle spinner's resource cost. Add `parallel` to the selector
   only when prepared and qualified, with explicit worker cpuset/priority
   configuration. Do not choose a CPU by proximity alone. Any cpuset changes
   follow the operator's reassignment decision and preserve EDT IRQ routing.

5. **Qualify, document and retain only useful changes.** Perform local
   correctness/lifecycle checks, then provide the operator with the concrete
   RTC A/B procedure. Record results and resource cost here before proposing
   any default change. Apply full-file documentation and clang-format to
   touched C/C++ files; restore `//mutex scope` comments afterward. Keep
   functional changes with their engineering record, documentation changes,
   and formatting-only cleanup in the separate commits required by AGENTS.md.

Expected implementation files are a small OCAM helper, standalone benchmark,
new real-SDK tests, `apps/ocam2KCtrl/ocam2KCtrl.hpp`, the relevant existing
OCAM tests, test registration/build support as needed, and the operator guide.
`libMagAOX` and RTC cpuset scripts change only if the investigation establishes
a shared improvement or an approved worker placement, respectively.

## Follow-up and Edge Cases

- Current digital binning repeatedly assigns rather than sums X samples,
  ultimately keeping the last sample; its loop does not apply Y binning.
  Preserve this behavior and discuss any correction separately. It must not
  become an accidental image change in a performance patch.
- The SDK stores a global table, reuses it while cameras remain initialized,
  and does not robustly reject malformed counts/indices. A synthetic-map
  extraction assumes a valid loaded table; checking its output cannot prevent
  an out-of-range access that already occurred inside the SDK. SDK parser
  hardening is a separate scoped follow-up, not a claimed benefit here.
- Handle mode changes and SDK-handle reuse, including valid handle zero.
  Quiesce worker jobs before SDK release or map/storage replacement.
- A two-thread win may disappear after cross-core staging reads, may worsen
  tails, or may require moving a task whose new placement hurts the loop.
  Report complete-system results and resource changes, not just half-kernel
  duration. The same-core implementation remains the fallback.
- No new generic library abstraction is justified yet. Existing staging,
  streamCopy, thread startup and timing are the reusable pieces identified.
- The initial planning pass changed only this document. Implementation and
  local measurements following approval are recorded below; instrument
  qualification remains an operator task.


## Single-core execution record — 2026-11-16

Implemented on `jrmales/ocam-descramble`, following the approved first three
phases. No worker thread, CPU reassignment, instrument commands, or deployed
configuration changes are included.

### Final implementation decisions

- `ocamDescrambler.hpp` derives the map once through the unchanged SDK and
  validates all source indices. It can represent all four app SDK modes for
  offline comparisons. The app exposes the fast path only for structured
  bin2, leaving all other modes and unmatched maps on the SDK.
- The selected kernel stores four source bases per output row and four
  validated runtime strides. It performs the same 14,400 word copies while
  removing per-pixel index-table loads. Every 30-pixel run is checked against
  the configured map before this kernel is enabled.
- A compact `uint16_t` map was faster than the SDK, but the first structured
  kernel was slower than that compact map on this workstation. Assembly
  inspection showed GCC expanding compile-time stride-16 loads into a large
  shuffle sequence. Retaining strides derived from the map produced a leaner
  scalar kernel and the best measured result. No SIMD intrinsics, compiler
  optimization pragmas, prefetch, extra frame copy, or shared library change
  was needed.
- `camera.descrambleMethod` defaults to `sdk`; `optimized` is opt-in.
  `descramble_method` is the live INDI request, and `descramble_status.applied`
  acknowledges the method used by a completed load. `availability` reports
  `ready`, `unsupported mode`, `unavailable table`, or `unconfigured`.
  Unsupported requests retain SDK conversion and report Alert; they remain
  requested so a later compatible configuration can apply them.
- The frame thread reads the request once, completes conversion and any
  existing digital sampling, then changes the applied-method snapshot only
  when needed. INDI updates run on the ordinary thread. The control snapshot
  group is cache-line separated from the changing frame counters to avoid
  adding false sharing to the SDK baseline. Staging, metadata ordering,
  frame validation, rearm, and camera serial control remain unchanged.
- The fixture generator reproduces all 57,600 entries in the local table;
  ordinary tests do not depend on `/opt/MagAOX/config`. An optional
  `MAGAOX_OCAM_DESCRAMBLE_TABLE` environment variable adds actual-table checks.
  The SDK's pre-existing malformed-table handling is not changed.

### Local benchmark evidence

Machine: `mx11`, Intel Core i9-12900HK, Linux CPU 2 affinity, ordinary scheduler.
Compiler: GCC/G++ 15.2.0, `-O3 -march=native -ffast-math`, C++20. Table identity
is the SHA-256 recorded above; benchmark FNV-1a64 is `ebf98bbf036f9adb`.
Three bin2 runs each collect 70,000 samples per algorithm/path/input population
(seven rounds of 10,000 samples), rotating comparison order in 128-frame blocks.
Each path is warmed up with 1,024 calls. All generated inputs are checked against
the SDK before timing. Timer median was about 0.014 us and is not subtracted.

Representative run 2, all durations in microseconds:

| Input / path | Method | Mean | Median | Std. dev. | p99 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| One hot frame / kernel | SDK | 5.2616 | 5.2140 | 0.4877 | 6.6410 | 32.9100 |
| One hot frame / kernel | Optimized | 2.8479 | 2.8120 | 0.3052 | 3.3930 | 26.0240 |
| 64 raw buffers / kernel | SDK | 5.4194 | 5.3000 | 0.8544 | 7.5410 | 35.9130 |
| 64 raw buffers / kernel | Optimized | 3.1277 | 3.0590 | 0.6808 | 4.5610 | 37.9470 |
| 64 raw buffers / kernel + streamCopy | SDK | 6.2400 | 6.0630 | 1.4976 | 8.6480 | 43.4160 |
| 64 raw buffers / kernel + streamCopy | Compact map | 4.6137 | 4.4490 | 1.4455 | 8.0080 | 52.3370 |
| 64 raw buffers / kernel + streamCopy | Optimized | 4.0022 | 3.8740 | 1.3297 | 6.3230 | 36.7620 |

The selected kernel improves mean time in all 84 paired round comparisons
(three runs, seven rounds, two input populations, two copy paths), saving
2.21–2.78 us. The representative staged 64-buffer mean improves by about 36%.
Run 3 is slower across methods, but the advantage persists. Workstation
outliers remain; the optimized kernel's maximum is not always lower.
These results qualify a candidate for testing, not RTC latency or tail behavior.
They do not predict how much of the reported 37 us interval will be removed.

In run 2, raw/private/published addresses modulo 64 are 32/48/16 for the hot
population and 16/32/48 for the rotating population. Raw frame stride is a
multiple of 64. There is no EDT DMA, shared-memory reader, or real-time thread
in this benchmark. Raw logs are
`/tmp/ocamDescramble-bin2-production-{1,2,3}.csv` on the development host;
the reproducible benchmark is committed with the implementation.

### Verification and remaining qualification

Local verification:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| Real SDK, generated and installed table | 1,020 | 4 |
| Same SDK suite with ASan + UBSan | 1,020 | 4 |
| New application selection/fallback/digital-bin tests | 7,327 | 5 |
| Existing OCAM main/EM gain/reconfig/lifecycle/temperature/housekeeping | 2,466 | 73 |
| frameGrabber integration | 150 | 8 |
| streamCopy | 34,704 | 1 |

Sanitizers report no bounds or undefined-behavior failures. LeakSanitizer is
not run because the sandbox's ptrace environment prevents it; ASan uses
`detect_leaks=0`. SDK-equivalence tests include non-affine valid tables,
source immutability, output guards, 16-bit patterns, alignment, handle zero,
clear/reprepare and independence from SDK table lifetime. App tests cover
pending/applied selection, malformed requests, whole-frame concurrent switches,
mode/table fallback, release/reprepare, inactive readback and nonuniform
software-bin behavior through both SDK and optimized dispatch.

The normal application builds and links against the copied EDT SDK with:

```bash
make -C apps/ocam2KCtrl EDT=true \
    EDT_INCLUDES=-I/home/jrmales/Source/opt/EDTpdv \
    'EDT_LIBS=-L/home/jrmales/Source/opt/EDTpdv -lpdv -lpthread -lm -ldl' -j1
```

The first parallel build exposed an existing PCH/schema-generation ordering
race (`telem_gpu_bfbs` / `telem_irq_bfbs` declarations not yet generated).
Regenerating logger outputs through the existing Makefile target and building
serially succeeds without tracked changes outside this task. A build-graph
fix is a separate follow-up. The SDK-linked app was never launched.

The operator must still run SDK/optimized/SDK comparisons on the RTC using
[the operator guide](../../../apps/ocam2KCtrl/doc/descramble-comparison.md),
keeping staging, readers and IRQ placement fixed.
No default change is proposed before that comparison. A helper-thread trial
is a separate next phase only if further improvement is worthwhile.


### Affected files

- `apps/ocam2KCtrl/ocam2KCtrl.hpp`: configuration, INDI request/status,
  preparation, frame-boundary dispatch and lifecycle invalidation.
- `apps/ocam2KCtrl/ocamDescrambler.hpp`: compact mapping and validated
  single-core bin2 kernel.
- `apps/ocam2KCtrl/Makefile`: helper header dependency.
- `apps/ocam2KCtrl/benchmarks/ocamDescramble_bench.cpp`, `Makefile`, and
  `.gitignore`: standalone comparison and build-output handling.
- `apps/ocam2KCtrl/tests/ocamDescrambleFixture.hpp`,
  `ocamDescrambler_test.cpp`, and `ocam2KCtrl_descramble_test.cpp`: deterministic
  mapping fixture, real-SDK equivalence and application regression coverage.
- `tests/Makefile.one`, `tests/tests.list`, and `tests/groups.dox`: isolated
  SDK linkage, test registration and application-unit-test documentation.
- `apps/ocam2KCtrl/doc/descramble-comparison.md`: configuration, status,
  operator RTC comparison and offline reproduction instructions.
- `agents/plans/2026-11/ocam_descramble.md`: approved decisions and execution
  evidence, preserving the original task text above the agent section.

Full-file declaration/member documentation review covered the touched C/C++
files, including correction of the existing Y-bin description without changing
its behavior. Functional changes, operator documentation, and formatting are
kept in separate commits. The plan's original protected template text has four
pre-existing trailing-whitespace lines; these are deliberately preserved.


## First operator RTC comparison — 2026-11-16 12:02 UTC

The operator supplied 23 `fgtimes` summaries spanning
12:02:15.350737034–12:02:47.984233007 UTC and identifies the downward step as
the switch to `optimized`. The supplied acquisition/write means remain near
500 us (approximately 2,000 Hz). The excerpt does not include applied-method
readback, deployed build identity, stage metrics, reader/IRQ settings, frame
quality counters, or an SDK return segment.

Exclude the apparent mixed transition window at 12:02:33.727964234 UTC:
its `wma` is 35.6045 us with 2.07548 us standard deviation. Average the 12
reported SDK summaries ending at 12:02:32.191924593 UTC and the 10 optimized
summaries starting at 12:02:35.263989137 UTC separately:

| Reported metric, us | SDK | Optimized |
| --- | ---: | ---: |
| Mean acquisition-to-write (`wma`) | 37.0655 | 34.0654 |
| Acquisition-to-write standard deviation | 1.11824 | 1.58492 |
| Write-interval standard deviation | 1.61142 | 1.99391 |

These are arithmetic averages of reported window statistics, not pooled
independent frame populations. Rolling windows can overlap, and the excerpt
has no sample counts or raw samples from which to infer percentiles.

The mean improves by **3.0001 us (8.09%)**. Acquisition-to-write standard
deviation increases by **0.46668 us (41.73%)**. The optimized standard deviations
remain 1.53874–1.61977 us, above the SDK range 1.08419–1.14939 us, after the
transition window has passed. This is a sustained mean/spread tradeoff in the
supplied excerpt; the no-jitter-regression acceptance criterion is not met.
It does not establish whether absolute high-percentile latency improved or
worsened, or prove the algorithm caused all of the change without a matched
return-to-SDK comparison.

The last two reports have lower acquisition-interval jitter (approximately
0.27 us) and arrive about one second apart, versus approximately 1.53 seconds
previously. Their cause is unspecified. Excluding those two still gives
optimized `wma` 34.0792 us and standard deviation 1.58674 us; acquisition-interval
standard deviation before this cadence change averages 0.70146 us versus
0.70316 us for SDK. Thus the observed `wma` spread increase is already present
before that later change in the acquisition/reporting statistics.

### Interpretation and next measurement

`fgtimes` reports actual standard deviations from frameGrabber's trailing
summary, including `acquisitionToWrite`. That interval starts at the SDK's
acquisition timestamp and ends at the stream write timestamp. It cannot
separate driver-to-thread delay, validation, descrambling or staging copy.

The optimized algorithm removes table loads **and changes access order**.
The SDK alternates the two output halves while its raw accesses advance inward
from opposite ends; the new structured kernel traverses successive output rows
and revisits the opposite tap lanes on a later reverse pass through the raw
rows. Cache behavior is therefore a plausible lead, not an established cause
of the measured standard-deviation increase. The selector's applied-status
atomic is only written on method changes, not every frame.

The next operator comparison should be SDK/optimized/SDK with identical
instrumentation and a full timing-window settling interval. With
`framegrabber.stageTiming=true` for all three runs, retain
`fg_stage_timing.conversion_{mean,stddev,min,max,count,valid}` and the associated
`waitToWrite`, `validation`, `timeoutQuery`, `writeBracket`, and quality
statistics. Conversion includes the metadata write flag, kernel, optional
digital sampling and staged copy. A variance change there localizes the issue
to that bracket; marginal variances must not be subtracted as independent
components without considering covariance. Include end-to-end loop timing and
frame failures/gaps before judging the overall tradeoff.

If conversion spread follows the selection, a controlled next candidate is a
compact map preserving the SDK's alternating-half access order. The existing
`mapped()` benchmark also traverses sequential output, so it does not isolate
map compaction from order. A second, targeted structured experiment would
process output row `y` immediately followed by `119-y`, retaining the current
four-stream loop for each. For the observed table, both consume the same raw
row, allowing immediate reuse. Neither experiment is implemented by this
analysis, and neither is claimed to reduce RTC jitter before measurement.

Only this plan is changed by the analysis. Parsed window values and arithmetic
are retained locally in `/tmp/ocam-descramble-rtc-20261116-summaries.csv` and
`/tmp/ocam-descramble-rtc-20261116-analysis.json`. No instrument command, source
change, default change, or push accompanies this update.


## Operator SDK/optimized/SDK stage comparison — 2026-11-16

The operator supplied three `fg_stage_timing` snapshots labeled `sdk`,
`descramble`, and `sdk`. Source attachment SHA-256:
`a5c12f4516d09a12a37759f11ce3c1c5719c55919504aeb7029829812a98e187`.
Snapshot wall-clock times and `fg_timing_quality` were not included. Each
snapshot contains 348 numeric elements; the middle label is interpreted as
optimized following the operator's comparison context.

All three acquisition intervals are approximately 276.116 us, or **3,621.67 Hz**,
with 7,244 / 7,243 / 7,243 per-frame observations. This is a different cadence
from the earlier 2,000 Hz excerpt and is analyzed separately. The same-frame
metrics below all report valid variance estimates.

| Metric, us | SDK before | Optimized | SDK return |
| --- | ---: | ---: | ---: |
| Acquisition-to-write mean | 37.7670 | 34.9512 | 37.8438 |
| Acquisition-to-write standard deviation | 1.1246 | 1.6975 | 1.1086 |
| Conversion mean | 16.4755 | 13.5300 | 16.5534 |
| Conversion standard deviation | 1.0966 | 1.6867 | 1.0782 |
| Conversion maximum | 21.1100 | 21.5800 | 21.9410 |
| SDK-return-to-write mean | 17.5570 | 14.7388 | 17.6426 |
| SDK-return-to-write standard deviation | 1.1022 | 1.6937 | 1.0889 |
| Acquisition-interval standard deviation | 0.3508 | 0.3574 | 0.3586 |
| SDK-return-interval standard deviation | 0.3167 | 0.3326 | 0.3548 |
| Publication-interval standard deviation | 1.4480 | 2.1819 | 1.4528 |
| Write-clock bracket mean | 0.6240 | 0.7411 | 0.6233 |
| Write-clock bracket standard deviation | 0.0894 | 0.1305 | 0.0974 |

The SDK return restores both the conversion mean and its lower standard
deviation. Relative to the arithmetic midpoint of the two SDK snapshots,
optimized conversion saves **2.9845 us (18.07%)**, while its standard deviation
increases **0.5993 us (55.11%)**. Acquisition-to-write saves **2.8542 us (7.55%)**
and its standard deviation increases from approximately 1.1166 to 1.6975 us.
These midpoint comparisons summarize the two reference observations; they are
not pooled distributions or confidence intervals.

This substantially strengthens the link between method selection and the
conversion-region tradeoff. Acquisition and SDK-return interval spreads remain
similar, while SDK-return-to-write spread tracks conversion. Validation,
timeout query and rearm spreads remain small. The write-clock bracket mean
rises by about 0.1175 us, accounting for most of the difference between the
conversion and acquisition-to-write mean improvements. Its spread increase is
small relative to conversion. Variances of separate stages are not subtracted
because their covariance is unavailable.

The measured conversion bracket still includes `md->write`, the kernel,
optional digital sampling and the staging copy; this does not isolate the
kernel alone. Conversion maxima (21.110 / 21.580 / 21.941 us) and
acquisition-to-write maxima (42.429 / 43.094 / 43.554 us) are comparable across
these snapshots. Greater standard deviation therefore does not establish worse
absolute tail latency. No per-frame distribution or p99 is supplied.

Each snapshot has zero observed serial-overlap duration and no completed
serial-duration samples. All conditional early/late populations have zero
samples and invalid estimates; these cannot be read as zero variability or
as proof of no frame-quality problems. Without `fg_timing_quality`, frame drops,
failed acquisitions and diagnostic losses are not independently qualified here.

Next local experiment: retain the validated four-amplifier kernel but process
output row `y` immediately followed by `119-y` before moving to the next pair.
The observed map sends both rows to one raw row; this tests whether immediate
reuse helps without adding threads, copies, or changing pixel mapping. Compare
against the current row-order kernel in a hardware-free benchmark first. Any
candidate remains opt-in and requires another operator comparison to establish
its RTC jitter behavior.

The attachment was parsed locally into
`/tmp/ocam-descramble-rtc-stages-20261116.json`; no INDI or other hardware
commands were executed by the agent.


### Local paired-row screening outcome

Tested the minimal paired-row traversal in isolated sources under
`/tmp/ocam-row-order-652sre5s`, without modifying repository implementation
files. Three alternating baseline/candidate pairs each collected 50,000
samples per path on CPU 2 with the same production compiler flags. Every
benchmark input was checked against the SDK before timing.

For 64 rotating raw buffers, candidate means improve by 0.060–0.075 us for the
kernel and 0.055–0.090 us with staged streamCopy; staged medians improve by
0.065–0.084 us. Hot-kernel mean changes range from a 0.001 us improvement to a
0.006 us regression. The small locality improvement is repeatable, but staged
standard deviations are mixed: 1.281→1.294, 1.199→1.197, and 1.218→1.268 us.
p99 is also mixed. This local experiment does not demonstrate a jitter fix
and cannot predict AMD RTC behavior.

The temporary candidate passes the complete helper suite with the installed
table: 1,020 assertions / four cases. Its minimal patch, sources, binaries and
six benchmark logs remain under the temporary directory above. No candidate
kernel or selector change is adopted in this analysis.

For causal isolation, the next useful controlled candidate is the compact
16-bit map with the SDK's alternating-half pixel order, compared against SDK
and the current structured kernel. That retains SDK source/destination order
while removing the strided 32-bit table access. It requires new local
validation/benchmarking and an operator RTC trial before any jitter claim.


## Follow-up conversion snapshots and compact control — 2026-11-16

The operator explicitly identifies the third of these new readings as SDK:

| Conversion metric, us | SDK before | Optimized | SDK return |
| --- | ---: | ---: | ---: |
| Mean | 16.5146285 | 13.45059075 | 16.5710280 |
| Standard deviation | 1.00254118 | 1.61073239 | 1.00720100 |
| Maximum | 20.388 | 19.287 | 20.649 |

Relative to the midpoint of the SDK summaries, optimized saves **3.09224 us
(18.69%)** while standard deviation increases **0.60586 us (60.29%)**. The SDK
return closely restores the first baseline. The optimized observed maximum
is lower than both SDK maxima in these windows; increased standard deviation
again does not imply a worse absolute upper tail. Counts, percentiles, validity,
quality counters, applied-method readback and timestamps are not supplied.
FPS is absent from this excerpt, so it is not assigned to the 2 kHz population;
the operator clarified that the preceding full stage snapshots followed a
restart at nominal 3,600 FPS.

### Controlled implementation

Added an opt-in `compact` method alongside `sdk` and the existing `optimized`.
It reads the prepared 16-bit map while alternating output pixels `i` and
`7200+i`, matching the bin2 SDK's source-level traversal. This provides a
control for the sequential-row/four-amplifier traversal of `optimized`.
Compiler scheduling and the removal of SDK dispatch/frame-number extraction
remain differences; this is not a guarantee of identical machine access order
or a complete one-variable cache experiment.

The existing `mapped()` helper and benchmark retain their sequential-output
traversal. `compact()` falls back to that traversal for non-bin2 helper modes.
The app deliberately gates both candidates on the same validated structured
bin2 map, retaining SDK fallback for unsupported modes or unrecognized tables.
The request is latched once per frame; applied status changes only after the
image load and optional digital sampling complete; the existing frameGrabber
staging copy follows. SDK remains default.
No allocation, new thread, camera command, staging change or timing
instrumentation is added to the frame path.

Extended the real-SDK equivalence tests, runtime selector/fallback tests and
offline benchmark for this control. The operator procedure is now
SDK/compact/SDK/optimized/SDK with fixed FPS verified after restart, identical
stage timing, full-window settling, conversion and acquisition-to-write
statistics, frame-quality counters and end-to-end loop timing. The candidate
requires an operator RTC trial before any claim of lower jitter.

### Validation

- Real-SDK helper tests with the installed calibration table: **1,174 assertions
  / four cases**. This includes every supported helper mode, arbitrary and
  non-affine mappings, input immutability, destination guards and alignments,
  and conversion after release of SDK global state.
- ASan/UBSan helper build with the same table: **1,174 assertions / four cases**.
  LeakSanitizer remains disabled for the previously documented sandbox
  limitation; address and undefined-behavior checks are enabled.
- Runtime configuration/selection/fallback/digital-sampling suite: **14,648
  assertions / five cases**, including initial switch state, all directed
  method transitions, concurrent requests, ambiguous selections and retained
  SDK fallback for both candidates.
- Force-rebuilt existing main, reconfiguration and lifecycle suites:
  **902 / 36**, **379 / four**, and **seven / one** assertions / cases.
- Real EDT application compiled and linked successfully with serial make; it
  was not launched. No hardware, INDI or deployment commands were run.
- All five changed C++ files pass clang-format checks; the complete diff
  passes whitespace checks. Independent app/helper review found no correctness
  or fallback blocker.

The normal tests total **17,110 assertions / 50 cases**, excluding the repeated
sanitizer run. Logs are under `/tmp/ocam-compact-*-{build,test}.log` and
`/tmp/ocam-compact-sdk-build.log`; the instrumented helper executable is
`/tmp/ocam-descramble-sanitize/test`.

### Local benchmark

Measured after all test/application builds completed: GCC 15.2.0, production
`-O3 -march=native -ffast-math` flags, CPU 2 on mx11's Intel i9-12900HK,
installed table FNV-1a64 `ebf98bbf036f9adb`, seven rounds of 10,000 samples per
path with rotating method order. Every synthetic frame was checked against
the SDK before timing; no hardware was involved.

| 64 rotating raw buffers, us | Mean | Median | Standard deviation | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDK kernel | 5.1245 | 5.0570 | 0.4357 | 7.0170 | 39.0600 |
| Compact SDK-order kernel | 4.7564 | 4.6950 | 0.3691 | 6.5490 | 29.7840 |
| Sequential mapped kernel | 3.5251 | 3.4840 | 0.4150 | 4.8170 | 32.3110 |
| Structured optimized kernel | 2.9628 | 2.9290 | 0.3225 | 3.3530 | 27.8050 |
| SDK plus staged streamCopy | 5.9115 | 5.7740 | 1.4683 | 8.0660 | 37.3730 |
| Compact plus staged streamCopy | 5.5001 | 5.3730 | 1.4718 | 7.5470 | 36.9290 |
| Sequential mapped plus staged streamCopy | 4.3407 | 4.2320 | 1.3255 | 6.1790 | 35.2400 |
| Structured optimized plus staged streamCopy | 3.7807 | 3.6760 | 1.2726 | 5.4670 | 34.1870 |

Across hot/64-buffer and kernel/staged populations, all 28 paired round means
favor compact over SDK by 0.342–0.434 us. In the 64-buffer case, its mean gain
is 0.3681 us (7.18%) for the kernel and 0.4114 us (6.96%) with staged copy.
Its staged standard deviation is close to the SDK's in this local run. The
structured path remains faster here. These synthetic workstation results
justify exposing compact as an opt-in comparison; they do not demonstrate an
RTC jitter fix or reproduce DMA and cross-core shared-stream readers.

The CSV includes hot-input statistics, all round means, timer overhead and
buffer alignments: `/tmp/ocamDescramble-compact-bin2.csv`. The mean ordering is
consistent within this run; no further repetitions were needed to establish
its role as a control. RTC qualification remains the operator's next step.

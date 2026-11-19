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
Status as of 2026-11-18: **the operator's sleep-then-spin parallel trials
retain approximately 29 us acquisition-to-write time at both 2 and 3.6 kHz,
without the large interval-jitter penalty seen during continuous spinning**.
With the configured 100 us post-job sleep, the 2 kHz WMA means average
29.003 us versus the earlier optimized 33.760 us; reported WMA standard
deviation averages 0.775 us versus 1.067 us. At 3.6 kHz, eight WMA window
means average 28.861 us with 0.975 us reported standard deviation, and
acquisition/publication interval standard deviations average 0.591/1.367 us.
The higher-rate capture therefore retains low latency and ordinary interval
spread, although its WMA spread is 0.200 us above the 2 kHz sleep trial.

Short captures now cover the stated maximum rate. A matched optimized
reference at 3.6 kHz, longer-run tails/frame delivery and complete-loop checks
remain before broader qualification. The generic requested-method default
remains optimized; no new default or resource assignment follows automatically
from these short captures.

The RTC reports a 950 ms/s RT runtime budget. Continuous-spin parallel had
about 28 us acquisition/publication interval spread, which disappeared when
the operator switched back to optimized. Sleeping between jobs resolves that
symptom in this trial and is consistent with CPU-budget pressure; direct RT
throttling evidence has not been captured. The older 37.423 us capture labeled
parallel remains invalid worker evidence because an empty power target caused
SDK fallback with the worker parked. SDK remains explicitly selectable and
the fallback for unsupported modes or unvalidated mappings. Deployment and
hardware execution remain operator steps.

## Task Summary

Reduce camwfs's acquisition-to-publication cost without degrading image
correctness, jitter, frame delivery, or end-to-end AO latency. Following the
operator comparisons, default to the validated optimized single-thread path,
retain SDK as the explicit reference and unsupported-mode/table fallback, and
keep live INDI comparison methods. The corrected worker with 100 us post-job
sleep improves processing mean/spread without the large interval regression
seen during continuous spinning at 2 kHz. A short 3.6 kHz trial also retains
low latency without that cadence penalty. Complete matched-rate reference and
sustained loop comparisons before deciding whether to dedicate the second
core for normal operation.

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


## Operator compact-control result — 2026-11-18

The operator reports that `compact` is **a few hundred nanoseconds slower than
SDK, with the same observed jitter as SDK**. This is a qualitative comparison:
the report does not include raw summaries, timing-metric identity, FPS, sample
counts, applied-method snapshots, percentiles, or frame-quality counters.
Do not assign it to one of the earlier FPS populations or infer exact timings.

The local synthetic benchmark's modest compact-map mean gain did not carry
over to this RTC comparison. Compact therefore offers no demonstrated
operational advantage over SDK in the reported test; retain it as a diagnostic
control, with SDK remaining the default/reference. The earlier structured
`optimized` result remains a separate mean-versus-spread tradeoff, rather than
a qualified improvement under the no-jitter-regression criterion.

Matching SDK jitter while retaining its source-level pixel traversal supports
investigating traversal and generated memory-access code as contributors to
the structured kernel's extra spread. It weakens the explanation that the
common selector/applied-status path alone causes that increase. It does not
prove a cache mechanism: compact also changes table layout and compiler code,
and the reported metric may include the staged copy and its interaction with
the kernel. No conclusion about extreme latency or frame delivery follows
from the qualitative jitter comparison alone.

The narrowest next diagnostic is to expose the already tested `mapped()`
helper for a matched compact/mapped/compact operator comparison. Both use the
same prepared 16-bit index map, while mapped traverses output sequentially.
This avoids introducing another kernel and tests whether the extra spread
also occurs with sequential output rather than only with the structured
four-amplifier loop. Staging, instrumentation, FPS and reader workload must
remain fixed; generated code and copy interactions still prevent interpreting
this as a pure hardware cache experiment.

A subsequent performance candidate could remove per-pixel map loads while
retaining SDK traversal: for first-half row `y`, visit amplifier runs in order
and alternate each pixel with the same column in row `y+60`, using validated
row bases and strides. This differs from the earlier `y`/`119-y` experiment,
which deliberately changed SDK order. It would need offline validation and
an operator RTC comparison before a performance or jitter claim.

This update records the result and its interpretation only. Runtime code,
defaults and instrument state are unchanged.


## Expose mapped traversal for RTC comparison — 2026-11-18

Following the operator's approval, `mapped` is now a fourth accepted
`camera.descrambleMethod` value and `descramble_method` switch, alongside
`sdk`, `compact` and `optimized`. The frame path calls the existing
`ocamDescrambler::mapped()` implementation, using the same prepared 16-bit map
as compact but writing sequential output pixels. The helper and offline
benchmark algorithms are unchanged.

Mapped uses the same validated structured-bin2 eligibility as the other
candidates in the app. Unsupported modes and unrecognized tables retain SDK
fallback and Alert status. SDK remains the default. The request is latched
once per frame, applied status reports the completed load, and existing
digital sampling and subsequent staged copy retain their behavior.

The operator guide now describes compact/mapped/compact with method
acknowledgment and a full timing-window settling interval for each capture.
Use identical FPS, instrumentation, staging and reader workload; retain SDK as
the operational reference. The previous local mapped benchmarks remain useful
context but do not establish mapped's RTC speed or jitter.

Extended the selector tests to four methods, including initial readback,
transitions, concurrent requests, all 11 ambiguous switch combinations,
candidate fallback/lifecycle and digital sampling.

Validation:

- Four-choice selector suite: **21,989 assertions / five cases**.
- Force-rebuilt existing main, reconfiguration and lifecycle suites:
  **902 / 36**, **379 / four**, and **seven / one** assertions / cases.
- These integration/regression tests total **23,277 assertions / 46 cases**.
  The mapped helper's existing real-SDK equivalence tests and algorithm are
  unchanged; this update exercises its runtime integration.
- Both changed C++ files pass clang-format checks, and the diff passes
  whitespace checks. Independent review found no omitted configuration,
  selection, fallback, dispatch or applied-status handling.

The real-EDT application also compiled and linked successfully with serial
make. Normal logger prerequisites were regenerated; existing FlatBuffers
field-naming warnings were emitted. No unrelated tracked files changed.
The application was not launched and no instrument commands were executed.
Logs are under `/tmp/ocam-mapped-*-{build,test}.log` and
`/tmp/ocam-mapped-sdk-build.log`.

Mapped is ready for the operator's RTC comparison; no new RTC performance or
jitter result is claimed by this integration.


## Four-method operator timing comparison — 2026-11-18

The operator supplied five conversion snapshots, each accompanied by four
`fgtimes` summaries, spanning telemetry timestamps 19:54:38–19:58:50.
The first four blocks are labeled SDK, optimized, compact and mapped. The
operator subsequently confirmed that the initially unlabeled final block is
a return to SDK. Source attachment SHA-256:
`72441231fa28ccb6e5aeb560bd86e8aa32ab6e2f1b2b7359ed8e1f25e5ca8613`.

All conversion snapshots have **count=4,000 and valid=1**. Acquisition means
are approximately 500 us, establishing approximately **2,000 FPS** and
approximately two-second conversion populations. These are separate from the
earlier nominal 3,600-FPS run. The individual stage snapshots are not precisely
timestamped; the adjacent telemetry supports the cadence but does not prove
identical sample populations. Applied-method readbacks, frame-quality counters,
per-frame distributions and end-to-end loop measurements were not included.

| Conversion, us | Mean | Standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| SDK before | 14.798577 | 0.832616 | 11.602 | 17.493 |
| Optimized | 10.800893 | 0.935667 | 9.007 | 15.499 |
| Compact | 14.931164 | 0.937344 | 11.472 | 17.933 |
| Mapped | 11.579746 | 1.250348 | 8.817 | 16.211 |
| SDK return | 14.768896 | 0.854879 | 11.391 | 17.723 |

Average each block's four reported telemetry means and standard deviations
arithmetically for the table below. These are **not pooled distributions**:
the rolling windows overlap, and the four reports are not four independent
experiments. Standard deviations are not standard errors.

| Telemetry summaries, us | Acquisition-to-write mean | Acquisition-to-write standard deviation | Publication-interval standard deviation |
| --- | ---: | ---: | ---: |
| SDK before | 38.509216 | 0.950821 | 1.350043 |
| Optimized | 34.683247 | 1.024190 | 1.422580 |
| Compact | 39.007829 | 1.040668 | 1.436183 |
| Mapped | 35.363765 | 1.302363 | 1.821928 |
| SDK return | 38.652514 | 0.963074 | 1.288670 |

The SDK return restores conversion closely: its mean is 0.029681 us lower
and standard deviation 0.022263 us higher than the first SDK capture.
Acquisition-to-write mean drifts upward by 0.143297 us, so the full pipeline
baseline is not identical, but this drift is much smaller than the optimized
gain. Use arithmetic midpoints of the two SDK reference summaries below:
conversion mean/standard deviation 14.783736/0.843747 us and
acquisition-to-write mean/standard deviation 38.580865/0.956947 us. These
midpoints are descriptive averages, not pooled distributions or uncertainties.

- Optimized saves **3.982843 us (26.94%)** in conversion. Its standard
  deviation increases **0.091920 us (10.89%)**, and its observed maximum is
  **1.994–2.224 us below** the two SDK maxima. Acquisition-to-write improves
  **3.897618 us (10.10%)**; its average reported standard deviation increases
  **0.067243 us (7.03%)**.
- Compact is **0.147427 us slower** in conversion and **0.426964 us slower**
  acquisition-to-write. Its conversion standard deviation is 0.093596 us
  above SDK, approximately equal to optimized in this run. Thus this numeric
  capture refines, rather than exactly reproduces, the earlier qualitative
  report of matching SDK jitter.
- Mapped saves **3.203991 us (21.67%)** in conversion, but its standard
  deviation increases **0.406601 us (48.19%)**. It is **0.778853 us slower**
  than optimized, with **0.314681 us greater standard deviation** and a
  **0.712 us higher observed maximum**. Its acquisition-to-write mean is
  0.680517 us slower than optimized and its spread is also greater.

Optimized is the most promising replacement in this capture. Its jitter
penalty is much smaller than in the earlier runs, including the first 2 kHz
excerpt, but conditions and build history differ; this does not establish
that a particular change fixed jitter. SDK still has the lowest measured
standard deviation among labeled methods, so the strict no-jitter-regression
criterion is not met. Observed maxima favor optimized, but 4,000 samples and
summary statistics do not establish p99 or longer-run tail behavior.

The compact/mapped contrast strengthens the case that traversal and generated
memory-access code matter: the same compact map yields a large speed change
and larger spread when output traversal changes. It does not prove a cache
mechanism or attribute the variance exclusively to the kernel, because
conversion includes metadata handling and staged copy. Mapped's broader
acquisition-to-write spread persists across all four reports, not solely the
one report with a larger acquisition-interval standard deviation.

Further qualification should prioritize repeated matched SDK/optimized/SDK
captures at this verified FPS and workload, including frame delivery and
end-to-end loop timing. Mapped remains a diagnostic control rather than the
preferred performance candidate in this result. A second descrambling worker
had not yet been implemented or tested at this point; this comparison covers
single-thread conversion only.

Parsed values and arithmetic are retained in
`/tmp/ocam-descramble-rtc-20261118.json`. This update changes documentation only;
no instrument command or runtime modification was made.


## Operator comparison with stage timing disabled — 2026-11-18

The operator reconfigured with stage timing off and supplied five SDK
`fgtimes` reports at 20:30:04.431607705–20:30:08.433582739 and six optimized
reports at 20:30:16.967600434–20:30:22.502998826. Acquisition and publication
means remain approximately 500 us, or **2,000 FPS**. No return-to-SDK segment,
sample counts, extrema, frame-quality counters or end-to-end loop timing are
included in this excerpt. These are ordinary acquisition-to-write statistics;
they cannot provide a new conversion-stage measurement with stage timing off.

Arithmetic averages of all supplied window summaries, in microseconds:

| Metric | SDK | Optimized |
| --- | ---: | ---: |
| Acquisition-to-write mean | 37.727338 | 33.817002 |
| Acquisition-to-write standard deviation | 1.009026 | 1.067993 |
| Acquisition-interval standard deviation | 0.329715 | 0.421443 |
| Publication-interval standard deviation | 1.354402 | 1.470970 |

Optimized saves **3.910337 us (10.36%)**, with **0.058967 us (5.84%)** greater
reported acquisition-to-write standard deviation. Publication-interval
standard deviation increases 0.116568 us (8.61%). These are averages of
reported rolling-window statistics, not pooled distributions or independent
trials; no confidence interval or percentile is inferred from them.

The optimized acquisition-to-write standard deviations range from
1.04788–1.09477 us, above the SDK range of 0.987757–1.03165 us. The first
optimized mean is somewhat higher at 33.937264 us; retaining all reports
avoids assuming that it is a transition sample. Excluding it as a sensitivity
check would give 33.792949 us mean and 1.062638 us average standard deviation,
leaving the same small-penalty conclusion.

The optimized report at 20:30:20.502015567 has acquisition-interval standard
deviation 0.825282 us and follows a longer reporting interval. Its
acquisition-to-write standard deviation is the lowest optimized value,
1.04788 us. That isolated acquisition/reporting disturbance does not explain
the acquisition-to-write spread increase. Excluding this report leaves
optimized acquisition-interval standard deviation 0.340675 us, much closer to
SDK, while acquisition-to-write standard deviation remains 1.072016 us.

### Comparison with stage timing enabled

Using the earlier SDK endpoint midpoint and optimized summaries:

| Acquisition-to-write, us | Stage timing on | Stage timing off |
| --- | ---: | ---: |
| SDK mean | 38.580865 | 37.727338 |
| Optimized mean | 34.683247 | 33.817002 |
| Optimized mean saving | 3.897618 | 3.910337 |
| SDK standard deviation | 0.956947 | 1.009026 |
| Optimized standard deviation | 1.024190 | 1.067993 |
| Optimized standard-deviation increase | 0.067243 | 0.058967 |

The approximately 3.9 us mean gain and small spread penalty persist with the
extra stage instrumentation disabled. Enabled stage markers are therefore
not necessary for observing this better tradeoff, and disabling them does not
recreate the earlier approximately 42% acquisition-to-write spread increase.
It still does not identify why that earlier run behaved differently.

The off captures have means lower by 0.853527 us for SDK and 0.866246 us for
optimized, consistent with removing a common timing cost. They were taken
after reconfiguration at a different time, rather than in a controlled
on/off/on crossover, so these differences are not an isolated measurement of
instrumentation overhead. Both absolute standard deviations increase slightly
with stage timing off; the difference between methods shrinks by only
0.008276 us. This is not evidence that disabling stage timing fixed jitter.

The current single-thread optimized path remains the most promising candidate
for longer matched SDK/optimized/SDK qualification in the intended operating
configuration. A small positive spread difference remains in the supplied
reports; tail latency and complete-loop performance are not established here.
Parsed values are retained in `/tmp/ocam-descramble-stageoff-20261118.json`.
Only documentation changes accompany this analysis; no instrument commands
or runtime modifications were made.


## Local second-thread experiment — 2026-11-18

The operator requested: "ok. let's do a local test of a 2nd thread."
Implemented a benchmark-only persistent worker and ran it on the verified
personal workstation `mx11`. Production `ocamDescrambler`, the camera app,
runtime method choices and instrument configuration remain unchanged. No
camera, INDI, DMA or shared-memory stream was accessed.

### Implementation and correctness

`benchmarks/ocamDescrambleWorker.hpp` prepares its own immutable row bases and
strides through the existing validated helper's public mapping operation.
It mirrors the production four-amplifier row loop without changing the helper.
One persistent `std::thread` and the caller process disjoint output regions:

- `parallel_halves`: 60 output rows per thread. Both traverse overlapping raw
  working sets with the installed mapping.
- `parallel_paired28`: caller handles 28 matching `y`/`119-y` row pairs and
  worker handles 32. At a 64-byte-aligned output base, ownership boundaries
  are aligned and the observed map gives separate raw-row groups.
- `parallel_paired30`: 30 matching row pairs each. Work is balanced, but two
  output boundary cache lines are shared even at an aligned base.

Separate cache lines hold release/acquire job and completion counters. There
is no per-frame allocation, thread creation or logging. Each call returns only
after the worker has finished using both buffers. Optional caller/worker
clocks run in a separate diagnostic pass. Handoff-only timing measures a
mandatory empty job. The worker acknowledges blocking park before every
single-thread baseline, and park/resume are outside measured calls.

The unchanged production `optimized` path remains the principal reference.
`range_serial` uses the mirrored loop with ordinary output-row traversal;
`paired_serial` processes complementary rows consecutively on one thread.
These controls separate worker handoff from compiler/traversal differences.
All staged measurements include the primary thread's existing `streamCopy`
after completion; they therefore include cross-core private-buffer reads.

The standalone worker checks passed **1,162 checks** in each of the ordinary
optimized, AddressSanitizer plus UndefinedBehaviorSanitizer, and ThreadSanitizer
builds. Leak detection was disabled for the sandboxed ASan run. Checks cover
SDK pixel equivalence, multiple input patterns, guards and source immutability,
offsets 0/2/16, all partitions, repeated jobs and input reuse, park/resume,
invalid setup and affinity, owned mapping lifetime, and active/parked shutdown
after completed calls. Benchmark validation also compares every generated raw
frame against the real SDK at offsets 0/16/requested before timing. All six
performance invocations finished with valid output guards. Legacy CLI and SDK
validation passed in all four modes; diagnostics, pacing and invalid-argument
smoke checks passed. Changed C++ files were formatted and reviewed;
`git diff --check` passed.

This is a blocking, single-caller prototype. Concurrent destruction, camera
reconfiguration/recovery with outstanding work, and verified FIFO scheduling
remain requirements for any future runtime integration, not claims of these
offline tests. The existing app methods are still entirely single-threaded.

### Measurement setup

- Intel Core i9-12900HK, caller logical CPU 2 and worker CPU 4, verified separate
  P-cores with private L2 caches and shared L3. Their SMT siblings are 3 and 5.
  Only benchmark-thread affinity changed; ordinary `SCHED_OTHER`, priority 0.
- GCC 15.2.0, C++20, `-O3 -march=native -ffast-math -Wall -Wextra -pthread`;
  SDK C object uses `-O3 -march=native -ffast-math`.
- Local table `/opt/MagAOX/config/ocam2k/fli/ocam2_descrambling.txt`, SHA-256
  `477a1aa37cab3d9cd9b087cce3204675c6e45cbea904ec1a20e754a48602b324`;
  bin2, 32,736 raw samples and 14,400 output samples.
- Every invocation measures one hot raw buffer and a 64-frame ring. Methods
  rotate in 128-frame blocks across rounds, after 1,024 unpaced warmup calls
  per path. Output checksums are outside the timer. Baselines have a parked
  worker; parallel blocks have a spinning worker, including between paced
  frames. This difference is part of the proposed resource cost.
- Private and publication buffers use a 64-byte-aligned base plus offset
  0 or 16 bytes. Unpaced: 10,000 samples/round, seven rounds, two invocations
  per offset (order 0,16,16,0). Paced: 1,000 samples/round, three rounds,
  one invocation per offset, with `sleep_until` outside timing at 500 us.
  No builds, sanitizer runs or other agent benchmarks overlapped timing.

Exact trailing benchmark arguments after `TABLE` were:

```text
bin2 10000 64 2 7 4 0 1 0
bin2 10000 64 2 7 4 16 0 0
bin2 10000 64 2 7 4 16 0 0
bin2 10000 64 2 7 4 0 0 0
bin2 1000 64 2 3 4 0 0 500
bin2 1000 64 2 3 4 16 0 500
```

Raw CSV summaries, per-round means, metadata, hashes and parsed values are in
`/tmp/ocam-two-thread-20261118/`; sanitizer logs are in
`/tmp/ocam-worker-checks/`. These local temporary artifacts are not repository
files. Tables below report populations as measured, without pooling runs or
subtracting clock overhead.

### Tight-loop results

First aligned invocation, 64-buffer ring, **70,000 samples per path**. All
durations are microseconds; staged includes dispatch/wait where applicable
and the final caller-side copy.

| Method | Kernel mean | Staged mean | Staged standard deviation | Staged p99 | Staged maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production optimized | 2.9199 | 3.6415 | 1.1047 | 4.4200 | 55.7390 |
| Range serial control | 2.9148 | 3.6494 | 1.0305 | 4.6290 | 35.7400 |
| Paired serial control | 2.6357 | 3.3473 | 1.0395 | 4.0010 | 32.8410 |
| Parallel halves | 2.0642 | 4.8851 | 2.0728 | 8.8110 | 36.4940 |
| Parallel paired 28/32 | 1.7130 | 4.3149 | 1.8992 | 7.9050 | 37.3240 |
| Parallel paired 30/30 | 1.6306 | 3.7818 | 1.1947 | 5.4840 | 36.8600 |

The balanced two-thread kernel saves 1.2893 us (44.2%), but **loses 0.1403 us
(3.85%) after staging**. Its staged standard deviation and p99 are higher;
its smaller observed maximum alone does not establish better tails. Across
the four invocations, ring64 handoff-only means are 0.1672–0.1964 us.

Repeated ring64 staged means:

| Output offset / invocation | Production optimized | Paired serial | Parallel paired 30/30 |
| --- | ---: | ---: | ---: |
| 0 / first | 3.6415 | 3.3473 | 3.7818 |
| 16 / first | 3.7061 | 3.4133 | 3.9893 |
| 16 / repeat | 3.6770 | 3.3922 | 3.8042 |
| 0 / repeat | 3.7216 | 3.4290 | 4.0983 |

The balanced worker loses **0.1272–0.3767 us (3.46–10.12%)** after staging in
these four ring64 populations. Its p99 is higher in every one, although
standard deviations and maxima do not uniformly worsen. Hot-buffer totals
show the same mean ordering. The paired single-thread control instead saves
0.2848–0.2942 us (about 8%) in the ring64 staged path. That traversal is a
possible later single-core experiment, not a production change in this patch.

The separate diagnostic pass is consistent with ownership costs recurring
on the following frame: for ring64 paired30, the worker kernel mean rises
from 1.3747 us without staging to 2.3178 us with staging, while the caller's
kernel stays near 1.38 us. The caller reads worker-written lines during the
copy; the worker then writes those lines again on the next job. Diagnostics
use extra clocks and consecutive method blocks, so these numbers must not be
subtracted from headline totals to assign an exact cost. Cache ownership is
a plausible mechanism, not proven by these timings. Paired30 beats paired28
despite its shared boundary lines: boundary alignment alone does not predict
the total cost, and balancing work matters too.

### Sleeping at a nominal 2 kHz cadence

The result changes substantially when the caller sleeps between frames.
Ring64 staged populations below contain **3,000 samples per path**:

| Offset | Method | Mean | Median | Standard deviation | p99 | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Production optimized | 31.4250 | 40.0150 | 15.2147 | 53.2050 | 65.4890 |
| 0 | Paired serial | 29.0648 | 36.5800 | 14.5282 | 53.5110 | 63.4210 |
| 0 | Parallel paired 30/30 | 17.1059 | 20.9100 | 7.4924 | 25.1370 | 56.5790 |
| 16 | Production optimized | 35.8405 | 40.0780 | 11.0810 | 55.1450 | 109.0160 |
| 16 | Paired serial | 29.8911 | 36.6195 | 12.9680 | 49.8950 | 57.0510 |
| 16 | Parallel paired 30/30 | 16.5552 | 21.0640 | 7.8817 | 28.1460 | 45.4220 |

Here the worker clearly improves the measured staged populations, but all
methods are much slower and more variable than in tight loops. These clocks
measure conversion after waking, excluding the sleep and scheduling delay
before the start clock. Both cores
use `intel_pstate` with governor `powersave`, EPP `performance`, and allowed
frequencies from 400 MHz to 4.9/5.0 GHz. Frequency, power states and system
scheduling were not controlled or recorded per sample. Sleep/wake behavior,
cache residency and the extra spinning core can change the comparison;
the observed reversal cannot be assigned solely to any one of them.

Whole lost pacing slots across all 42 hot/ring method paths totaled 132 at
offset0 (0–14 per path) and 313 at offset16 (0–33 per path). Those are late
synthetic sample starts under ordinary scheduling, not camera dropped frames.
No catch-up bursts are intentionally issued after losing whole periods.
Sleeping synthetically is not a reproduction of EDT DMA or the RTC's power,
FIFO, AMD CCD and shared-reader conditions.

### Decision and remaining qualification

The local test establishes correctness and a substantial two-thread kernel
speedup. It also shows why kernel-only timing is insufficient: staging erases
that gain in tight loops, while a sleeping workload reverses the result.
Neither case establishes the RTC's acquisition-to-write gain or jitter.
Retain this as an offline experiment; keep the current production methods.
Before a runtime worker proposal, obtain a controlled same-CCD target-CPU
comparison including staging, actual placement and frequency conditions,
then qualify frame delivery and end-to-end loop latency with the operator.
A production implementation would additionally need framework thread startup,
verified scheduling, fault/recovery handling and configuration lifecycle tests.


## Runtime worker for operator RTC qualification — 2026-11-18

The operator requested an instrument comparison and offered to free a cpuset
on the framegrabber's CCD. Candidate placement is `spare31` (logical CPU 31)
if actual topology confirms shared L3 with `camwfs` (CPU 33); `camwfs_sw`
(CPU 32) is another proposed candidate. CPU numbers alone are insufficient.
The operator was initially asked for the RTC's SMT and L3-sharing lists.
The repository already contains an [operator-confirmed topology table](../2026-10/rtc-cpuset-restructuring.md#operator-confirmed-topology):
CPU 31 belongs to L3 domain 24–31, whereas CPUs 32 and 33 share L3 domain 32–39.
The current `rtSetup/RTC/make_cpusets` comments and the later
[camwfs shared-memory timing record](shmim-monitor-speedups.md#camwfs-avg-shmimintegrator-on-the-camera-stream)
agree with that placement. Therefore recommend `camwfs_sw` (CPU 32) after the
operator frees it; `spare31` does not satisfy the same-CCD condition. The app
rechecks live topology before admission. This change adds configuration and
verification, not a cpuset reassignment. The local
host remains `mx11`; no agent-side instrument commands or camera execution
are authorized or performed.

### Runtime integration

- Add `parallel` to `camera.descrambleMethod` and the live `descramble_method`
  selector, retaining SDK as default. New `descramble.cpuset` defaults empty
  and disables the worker. `descramble.threadPrio=-1` inherits the configured
  framegrabber priority; an enabled worker requires priorities 1–99.
  `descramble.partition` selects `paired30` (default), `paired28`, or `halves`.
- Start one persistent worker with `MagAOXApp::threadStart`, including the
  standard `th-descramble` property. The legacy startup boolean is not an
  admission barrier: the worker remains parked until actual policy, priority,
  exact configured single-CPU masks, distinct physical cores and shared L3
  have been verified. This follows the existing BMC scheduling-verification
  precedent because `threadStart` can return success after a scheduler error.
- Require `framegrabber.staged=true`, a prepared structured bin2 map and
  verified worker placement. Recheck placement during acquisition
  configuration. Failures retain SDK fallback and report an Alert for an
  unmet parallel request. Existing prepared single-thread choices remain
  usable. `descramble_status` adds parallel state and verified CPU readbacks.
- The frame thread latches one method for an entire load, completes both
  private output portions before existing digital sampling, and returns only
  when the worker has released the raw and output buffers. The original
  primary-thread copy, metadata/write-flag ordering, publications and EDT
  buffer-replenishment ordering remain unchanged.
- Switching to a single-thread method parks the helper. Power-off/control
  paths request parking without waiting for frame work under app locks;
  frame-owned recovery/configuration paths drain and park it. Shutdown joins
  the framegrabber, stops/joins the helper, then releases prepared maps and SDK
  state. Worker mode/map lifetime is separate from numeric SDK-handle reuse.

### Shared worker and placement support

`ocamDescrambleParallel.hpp` holds the prepared row kernel and mandatory-job
protocol without owning a thread, SDK handle or CPU configuration. Preparation
copies validated mapping data via the existing helper's public interface;
the production `optimized` implementation remains unchanged. The benchmark
thread/affinity wrapper now uses this same engine, so future local measurements
exercise the runtime handoff protocol rather than a separate prototype.

An atomic control word closes admission when park/stop is requested. A job
admitted concurrently with a control request must drain before the caller
returns or storage is reused. Release/acquire generation counters separate
job payload publication from completion. Atomic wait/notify handles cold
park/resume; the active job/completion path spins. The admission protocol adds
atomic operations compared with the original offline prototype, so earlier
microbenchmarks are historical results, not exact timings of this revision.
Unexpected or requested worker exit cannot authorize fallback while worker
writes remain outstanding. One owner invokes prepare/clear/resume/run; control
threads may request park/stop. Destruction still requires stop and join.

`ocamDescramblePlacement.hpp` reads scheduling, affinity, cpuset files and
kernel topology without changing them. Verification establishes observed
placement, not exclusive ownership or protection against later reassignment.
The operator must reserve the physical core, avoid conflicting FIFO tasks
there and on its SMT sibling, and retain that reservation through all test
intervals. The existing framework startup and staging utilities are reused.
BMC and OCAM now both need post-start scheduling/affinity verification; a
follow-up common utility could consolidate that check while retaining OCAM's
app-specific distinct-core/shared-L3 requirement. This scoped experiment does
not change BMC or the framework thread-start contract.

### Operator comparison

The comparison guide contains configuration and exact readbacks for
SDK → optimized → parallel → optimized → SDK, settling at least one full
rolling timing window after each acknowledgment. Keep the freed-core workload,
FPS, synchronization, gain, staging, readers and stage instrumentation fixed
through the crossover. Use applied-method, worker-running/parked and CPU
readbacks to confirm each condition. Measure conversion where enabled,
acquisition-to-write, frame quality and end-to-end loop timing. The operator
runs installation, restart, cpuset and INDI commands. No RTC performance
result is claimed by local code validation.


### Completed local validation

The production app compiled and linked successfully with the locally installed
real EDT SDK using:

```bash
make -C apps/ocam2KCtrl EDT=true \
  EDT_INCLUDES=-I/home/jrmales/Source/opt/EDTpdv \
  'EDT_LIBS=-L/home/jrmales/Source/opt/EDTpdv -lpdv -lpthread -lm -ldl' -j1
```

The default build path `/opt/EDTpdv` is absent on this workstation; the explicit
local SDK path above resolves that dependency. The built camera executable
was not launched or installed. Final build log:
`/tmp/ocam-runtime-app-build-edt-final.log`.

Forced-rebuilt offline app suites passed **31,965 assertions in 81 cases**:

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| Descramble selection/lifecycle | 29,499 | 8 |
| Main OCAM controller | 902 | 36 |
| Reconfiguration | 379 | 4 |
| Application lifecycle | 7 | 1 |
| EM gain | 34 | 1 |
| Temperature | 504 | 5 |
| Housekeeping | 640 | 26 |

The shared worker's standalone real-SDK suite passed **1,493 checks** in each
of the optimized, ASan/UBSan and TSan builds. Each run observed 48 admitted-job
park/stop races; no sanitizer diagnostic occurred. Admission observation
exercises real concurrent interleavings without adding a production delay
hook; it does not exhaust all possible schedules. The original pinned
benchmark wrapper retained its **1,162 passing checks**, and worker-diagnostic
and legacy benchmark smoke invocations validated output and guards.

The placement suite passed **150 assertions in three cases**, both normally
and with ASan/UBSan. It covers exact/wide/wrong affinity, invalid or absent
cpusets, policy/priority mismatch, SMT siblings, different L3/package,
malformed/missing topology, and reset output IDs after failure. Its only
placement mutations pin its own ordinary local test threads. ASan leak
checking was disabled because LeakSanitizer cannot operate under the tool's
ptrace environment; address, undefined-behavior and thread checks did run.
Logs and smoke outputs are in `/tmp/ocam-runtime-checks/`.

Independent review found and corrected a premature `parked` readback: a park
request now reports `parking` until acknowledged quiescence. SDK fallback
also invalidates worker availability after definitive worker exit, reporting
Alert rather than leaving an indefinitely Busy parallel request. C++ files
match clang-format with restored mutex-scope comments; the protected original
plan prefix is unchanged and `git diff --check` passes.

### Affected files and remaining operator steps

- `apps/ocam2KCtrl/ocam2KCtrl.hpp` and `apps/ocam2KCtrl/Makefile`.
- New `apps/ocam2KCtrl/ocamDescrambleParallel.hpp` and
  `apps/ocam2KCtrl/ocamDescramblePlacement.hpp`.
- `apps/ocam2KCtrl/benchmarks/ocamDescrambleWorker.hpp`, `Makefile`, `.gitignore`,
  and new `ocamDescrambleParallel_test.cpp` in that directory.
- `apps/ocam2KCtrl/tests/ocam2KCtrl_descramble_test.cpp` and new
  `apps/ocam2KCtrl/tests/ocamDescramblePlacement_test.cpp`.
- `tests/Makefile.one`, `tests/tests.list`, this plan, and
  `apps/ocam2KCtrl/doc/descramble-comparison.md`.

Local implementation and checks are complete. Reserving `camwfs_sw`/CPU 32,
deployment/restart, checking actual placement readbacks and the matched
measurement crossover remain operator steps. No cpuset assignment was changed
on the instrument, and no instrument performance result has yet been obtained
for `parallel`.


## RTC parallel comparison and optimized default — 2026-11-18

The operator supplied five optimized `fgtimes` reports with log timestamps
2026-11-19T00:40:30.376646582–00:40:34.379166168 and six reports labeled parallel
at 00:40:45.914394889–00:40:51.448416061. This entry uses the client-local date
2026-11-18. Acquisition and publication means remain approximately 500 us,
or **2,000 FPS**. The operator called the result clear and proposed making
optimized the default unless another experiment was warranted.

Earlier in this session the operator directly confirmed current CPU topology:
CPU31 has SMT list `31` and L3 list `24-31`; CPU32 has `32` and `32-39`; CPU33
has `33` and `32-39`. Thus CPU32/33 share L3 and have no other online SMT
siblings in that readback. CPU31 is not on their CCD. The comparison was
prepared with `camwfs_sw`/CPU32 for the worker and `camwfs`/CPU33 for acquisition.

The operator initially confirmed an active worker, then corrected that answer:
`parallel=parked`. A fresh readback while parallel was requested showed
`applied=none`, `availability=ready`, `caller_cpu=33`, `worker_cpu=32`, and
`parallel=parked`. Subsequent diagnostics confirmed FSM `OPERATING`, fresh
`fgtimes`, observed outlet state `On`, and an empty outlet target. The code
incorrectly treats that unknown target as a reason to use SDK fallback, park
the worker, and hide the applied method. The active-parallel attribution
previously recorded here is withdrawn. These windows do not establish parallel
performance; their historical method was not captured reliably.

The default decision relies on the earlier SDK/optimized/SDK comparisons.
No return-to-optimized segment, stage-timing setting, conversion-stage samples,
frame-quality counters or end-to-end loop measurements are supplied in this
excerpt.

Arithmetic averages of every supplied window summary, in microseconds:

| Metric | Optimized | Labeled parallel; active worker unconfirmed |
| --- | ---: | ---: |
| Acquisition-to-write mean | 33.760375 | 37.422939 |
| Acquisition-to-write reported standard deviation | 1.067152 | 1.118703 |
| Acquisition-interval reported standard deviation | 0.559981 | 0.587159 |
| Publication-interval reported standard deviation | 1.703262 | 1.770800 |

The second segment mean is **3.662563 us (10.85%) slower** than optimized,
with **0.051551 us (4.83%) greater reported standard deviation**. Publication-
interval standard deviation increases 0.067538 us (3.97%). These are arithmetic
averages of rolling-window summaries, not pooled frame distributions,
independent experiments, standard errors or percentile estimates.

All second-segment means, 37.400656–37.439290 us, exceed all optimized means,
33.714670–33.790704 us. Reported WMA standard deviations likewise do not overlap:
1.08049–1.18351 us versus 1.04968–1.07959 us. The first two second-segment
reports have larger spread, but excluding both as a sensitivity check gives 37.426396 us
mean and 1.097645 us average standard deviation; the slower-mean conclusion
persists. No samples are excluded from the main comparison.

The numerical difference remains valid for the supplied summaries, but it
cannot be attributed to parallel execution. The known power-target bug explains
how selecting parallel can instead measure SDK fallback. Repeat the comparison
after the fix, with `applied=parallel` and `parallel=running` verified before and
after the capture; placement alone does not demonstrate worker participation.

### Default decision and implementation

Accept the operator's proposed `optimized` default. Earlier matched RTC
captures found approximately 3.9 us improvement over SDK, with a small positive
reported spread difference that remains an accepted tradeoff rather than a
claim that the original strict no-jitter-regression criterion was met.
No further kernel experiment is required before this default change.

One optional future experiment is paired-row traversal on a single core: the
local control saved about 0.29 us including staging. It has no RTC measurement
and is a lower-priority follow-up, not an additional runtime method or a
reason to postpone this choice. Retain existing SDK/compact/mapped/parallel
choices for explicit comparison and recovery.

The patch changes the constructor's requested method, the omitted
`camera.descrambleMethod` configuration value, and configuration help to
`optimized`. Existing configuration/whole-frame tests now expect the missing
key and an initial prepared frame to select optimized. Explicit `sdk` and
other method settings still override the default; fallback, digital sampling,
worker parking and stream publication code are unchanged.

The existing status contract is retained: when optimized is requested in an
unsupported mode or with an unvalidated table, conversion uses SDK and the
unmet request reports Alert. An explicit SDK configuration avoids that Alert
for intentionally unsupported modes. No silent change to request/readback
semantics is included in this default-only patch.

The operator should remove any explicit `camera.descrambleMethod=sdk` override
or set it to `optimized` when adopting the new default. Removing/emptying the
experimental `descramble.cpuset` setting at the next restart after completing
the corrected parallel experiment avoids creating an unused worker and allows
the reserved core's previous workload to be restored. Keep any workload
restoration outside a matched timing crossover.
The assistant does not alter instrument configuration or run these steps.

Parsed supplied values and arithmetic are retained in
`/tmp/ocam-parallel-rtc-20261118.json`. Affected files are the controller header,
its existing descramble-selection test, this plan and the comparison guide.


### Default-change validation

The production app compiled and linked with the local real EDT SDK using the
explicit SDK include/library paths recorded above. Forced rebuilds of the
existing descramble-selection and main controller suites passed **29,499
assertions in eight cases** and **902 assertions in 36 cases**, respectively:
**30,401 assertions in 44 cases** total. The default config and initial
prepared frame now exercise optimized, while explicit SDK selection and
unsupported-mode/table fallback remain covered. No instrument executable was
run or configuration changed.

Build/test logs are `/tmp/ocam-default-app-build.log` and
`/tmp/ocam-default-20261118/`. The changed C++ files were formatted, their
full-file documentation remains consistent, the original plan prefix is
unchanged, and `git diff --check` passes.


## Unknown outlet target blocked parallel — 2026-11-18

During follow-up to the corrected `parallel=parked` report, the operator
confirmed that camwfs continued acquiring and publishing fresh `fgtimes` with
FSM `OPERATING`. The worker was present in `camwfs_sw/tasks`; placement readbacks
were caller CPU33 and worker CPU32. The power source reported
`pdu1.camwfs.state=On` with `pdu1.camwfs.target` empty.

An empty target is represented by `powerStateTarget() == -1`. This is a normal
reachable state: the outlet controller initially creates an empty target and
ordinary state polling does not populate it. The framework can acquire while
observed power is On, without a known target. Three descramble checks were
inconsistent with that behavior:

- Parallel admission rejected targets `<= 0`, so valid frames used SDK fallback.
- Post-publication handling requested worker parking for targets `<= 0`.
- Applied-method status required target `== 1`, masking completed conversions
  as `none` even while frames continued.

The fix treats only explicit target Off (`0`) as a blocking target in those
three checks. Observed power must still be On. Shutdown, reconfiguration,
staging, validated-map, and verified-placement gates are preserved. No power
commands, subscription behavior, serial housekeeping policy, or framegrabber
publication ordering change.

The offline regression holds actual power On and changes the target between
On, unknown, and Off, and separately checks actual Off with an unknown target.
It verifies post-publication worker state, next-frame pixel output against a
distinct SDK sentinel, selected/applied method, and worker status. Before the
fix, only the unknown-target case failed: five failed assertions reproduced
SDK output, `applied=none`, and a parked worker.
The On and explicit-Off cases passed. This reproduces the operator's symptoms
without running the camera or changing any instrument state.

Validation passed: the production app compiled and linked against the local
real EDT SDK. Forced rebuilds of the descramble-selection suite passed
**29,595 assertions in nine cases** and the main controller suite passed
**902 assertions in 36 cases**, totaling **30,497 assertions in 45 cases**.
The new regression passed all four state/target combinations. Build/test logs
are in `/tmp/ocam-parked-diagnosis/`. Changed C++ files were formatted and their
full-file documentation reviewed; the original plan prefix is preserved and
`git diff --check` passes. The hardware app was not executed.

The operator must deploy/restart and repeat the matched optimized/parallel/
optimized comparison, checking `applied=parallel`, `parallel=running`,
`availability=ready`, and CPUs 33/32 before and after the parallel interval.
No new parallel performance claim is made from the earlier capture.


## First confirmed parallel capture after the power-target fix — 2026-11-18

The operator reported pushing through `b11297d0`, then supplied seven
`fgtimes` reports as the follow-up to the parallel retest. Log timestamps run
from 2026-11-19T01:12:31.589383628 through 01:12:49.710186671; this entry uses
the client-local date 2026-11-18. The operator described a "big win". The
excerpt itself contains no method/worker readback, stage-timing setting,
adjacent optimized segment, or frame-delivery diagnostics. In follow-up, the
operator explicitly confirmed `applied=parallel`, `parallel=running`, caller
CPU33 and worker CPU32, and reported that the large acquisition jitter goes
away when switching back to optimized. This establishes an operator-observed
association with parallel mode; no numeric return-to-optimized excerpt was
provided, so the numerical comparison below still uses the earlier reference.

Arithmetic averages of the supplied window summaries, in microseconds:

| Metric | Earlier optimized capture | New reported parallel capture |
| --- | ---: | ---: |
| Acquisition-to-write mean | 33.760375 | 28.825956 |
| Acquisition-to-write reported standard deviation | 1.067152 | 0.799171 |
| Acquisition-interval reported standard deviation | 0.559981 | 27.797043 |
| Publication-interval reported standard deviation | 1.703262 | 27.827214 |

Relative to that earlier, unmatched optimized reference, acquisition-to-write
mean is **4.934419 us (14.62%) lower**, and reported WMA standard deviation is
**0.267981 us (25.11%) lower**. All seven WMA means are 28.532445–29.126591 us;
all reported WMA standard deviations are 0.758751–0.849137 us. These support a
substantial processing-path gain with the active method confirmed.
They remain averages of rolling-window summaries, not pooled frame statistics,
independent trials, tail measurements, or complete-loop latency.

Acquisition means are 500.369932–501.407725 us and publication means are
500.366743–501.399863 us, near 2,000 FPS. Their reported standard deviations
are respectively 20.7002–37.2242 us and 20.7779–37.2580 us, much larger than
the earlier reference and closely tracking one another. The added interval
variation is already present in the acquisition timestamps of published
frames; the WMA data do not show more variable acquisition-to-write processing.
These aggregate values do not distinguish camera/trigger cadence, acquisition
scheduling, or gaps between accepted/published frames. The operator's return
to optimized ties the effect to parallel mode, but does not isolate its cause.
The relation between successive published frames is
`writeInterval = acquisitionInterval + currentWMA - previousWMA`; similar
acquisition/publication spread is therefore compatible with nearly constant
per-frame write-minus-acquisition latency. It does not rule out an effect of
the worker on later acquisition or rearm.

Retain the optimized default and select optimized during diagnosis. Inspect
ordinary `fg_timing` acquisition/publication interval extrema and the camera's
skipped-frame, acquisition-failure, and frame-number-corruption logs if needed.
Optional `fg_timing_quality` counters are only exposed with stage timing enabled; do
not change instrumentation midway through a crossover.

### Targeted follow-up

Source review confirms the active worker continuously polls its control word
with a processor pause instruction between frames. That instruction does not
block the thread. Switching to optimized parks the worker, so the crossover
changes both pixel processing and continuous FIFO CPU occupancy. Idle polling
reads a separate control cache line; it does not repeatedly write the stream
or traverse the image while idle. This identifies a testable difference,
not proof of the mechanism behind the acquisition jitter.

First ask the operator for live `sched_rt_runtime_us`/`sched_rt_period_us` and
whether `camwfs_sw/tasks` contains only the descramble thread. These checks
can identify an RT-runtime limit or shared-core workload before changing the
protocol. No scheduling limits or instrument assignments are changed by the
assistant.

If those checks do not explain the effect, the next focused experiment is an
optional worker wait between jobs, waking when the owner publishes the next
job, while retaining the existing short completion wait and mandatory job
drain. Keep partition, core placement, staging, and timing settings fixed.
Do not implement this as per-frame park/resume in the post-publication hook:
that would add another acknowledgment before EDT rearm and confound the test.
Blocking idle would trade interframe CPU occupancy for wake-up latency; both
WMA and frame-interval distributions must be remeasured. This experiment has
not yet been implemented or tested.

Parsed values and arithmetic are retained in
`/tmp/ocam-parallel-fixed-rtc-20261118.json`. Only this engineering record is
changed; no runtime settings or implementation changed, and no instrument
commands were run by the assistant.


## Sleep then spin worker experiment — 2026-11-18

The operator reported RTC `kernel.sched_rt_runtime_us=950000` and
`kernel.sched_rt_period_us=1000000`, with only the descramble task in
`camwfs_sw/tasks`. The operator proposed sleeping for 100 us between frames,
then spinning, and specified a maximum camera rate of 3.6 kHz. This finite
95% RT budget makes continuous FIFO spinning a plausible contributor to the
observed cadence regression; it does not by itself prove that throttling
caused the supplied timing distributions.

At 2 kHz, a 100 us block per 500 us frame gives a nominal runnable fraction
near 80%; at 3.6 kHz, a 100 us block per 277.8 us frame gives about 64%.
Those estimates assume steady frames and timely wakeups. Sleep scheduling can
overshoot, and a worker that receives no new frames resumes spinning after
its one sleep. Thus this is a targeted 2–3.6 kHz experiment, not a general
RT-budget guarantee at low FPS or during acquisition stalls. No kernel RT
budget, scheduling priority, cpuset, or instrument configuration is modified.

### Implementation

- Add startup configuration `descramble.sleepUs`, default **100**, accepting
  integral values **0–200 us**. Zero preserves the continuous-spin control.
  The upper bound is conservative for this experiment: 200 us leaves only
  77.8 us of a nominal 3.6 kHz period, before accounting for job phase and
  scheduler oversleep. It is not a guaranteed wake deadline.
- Pass the immutable delay to the shared `workerMain(unsigned idleSleepUs=0)`
  entry point. The core's default remains zero so existing standalone wrapper
  and benchmark callers retain their previous behavior.
- After releasing `m_done`, the worker sleeps once, then returns to polling.
  There is no sleep before the first job. The completed frame's owner can
  proceed to staging/publication without waiting for the sleep. The worker
  does not access the completed frame's buffers after publishing completion.
- Use standard C++ `sleep_for`, already available in the project, with a
  microsecond duration. Skip the sleep if park/stop is already requested.
  A request arriving during sleep waits for that sleep and scheduler delay;
  a new admitted frame waits for wake and still drains before buffer reuse.
  No per-frame park/resume handshake or new owner wait is introduced.
- Expose the configured value as `descramble_status.sleep_us` and in the
  startup worker-verification log. This reports requested delay, not measured
  sleep duration. `parallel=running` continues to mean an active worker and
  includes its short post-job sleeps; it is distinct from administrative
  `parked` state.
- Preserve the optimized default, explicit worker cpuset requirement,
  mandatory job completion, map validation, digital sampling, staging,
  publication ordering, and acquisition/rearm behavior.

The application tests exercise default 100 us through the local worker
launcher, plus omitted/zero/100/200 configuration, readback and rejection of
negative, fractional, nonfinite and out-of-range values. Shared core tests
exercise zero and 100 us for full image correctness, guard preservation,
immediate buffer reuse, map replacement, park/resume, stop and admitted-job
control races. Functional assertions avoid hard scheduler-latency thresholds.
Validation passed:

- The real EDT application compiled and linked with the local SDK; it was not
  executed.
- Forced application rebuilds passed **29,730 assertions in ten descramble
  cases** and **902 assertions in 36 controller cases**: **30,632 assertions
  in 46 cases** total, including configured 100 us worker launches.
- Standalone checks passed **1,162 wrapper checks** with the original default
  and **3,116 shared-core checks** covering zero and 100 us. Each core run
  observed 48 admitted-job control races at each delay.
- The same 3,116 core checks passed under AddressSanitizer/UndefinedBehaviorSanitizer
  and ThreadSanitizer. Leak checking was disabled for the known tracing
  environment limitation; address/undefined-behavior checking remained enabled.
- Changed C++ files were formatted and their full-file documentation reviewed;
  the original user-owned plan prefix is preserved and `git diff --check`
  passes.

App logs are in `/tmp/ocam-idle-sleep-20261118/`; standalone logs are
`/tmp/ocam-runtime-checks/parallel-sleep-{check,asan,tsan}.log`. These checks
establish local functionality and protocol safety, not an RTC timing result.

### Operator comparison

After the normal operator deployment/restart, preserve CPU32/33 placement,
partition, FPS, synchronization, staged copy, reader workload and timing
settings. Confirm `sleep_us=100`, `applied=parallel`, `parallel=running`, and
intended CPUs during the parallel capture. Run settled optimized/parallel/
optimized windows at 2 kHz, then separately at 3.6 kHz if the 2 kHz result is
sound. Compare acquisition/publication interval distributions and frame
delivery as well as WMA; a lower WMA alone does not qualify the worker.

To reproduce continuous spin, set `descramble.sleepUs=0` and restart before
another matched crossover. The setting is startup-only; method changes remain
live. Keep the same scheduling budget for the comparison. A timer oversleep
can move a later frame's wake delay into WMA. The first 2 kHz instrument
result for the new sleep setting is recorded below.

Affected files: the controller and shared worker headers, their application
and standalone protocol tests, this plan, and the operator comparison guide.


## Sleep then spin RTC result at 2 kHz — 2026-11-18

The operator reported "that did it" and supplied ten `fgtimes` reports after
the 100 us sleep-then-spin change. Log timestamps span
2026-11-19T02:15:10.222825407–02:15:19.759283914; this entry uses the
client-local date 2026-11-18. The supplied acquisition/publication means are
500.000575/500.000507 us on average, confirming approximately **2,000 FPS**.
This is the operator-reported sleep trial; no separate status dump or new
numeric optimized reference is included with this excerpt.

Arithmetic averages of every supplied window summary, in microseconds:

| Metric | Earlier optimized | Continuous-spin parallel | Parallel with 100 us sleep |
| --- | ---: | ---: | ---: |
| Acquisition-to-write mean | 33.760375 | 28.825956 | 29.003462 |
| Acquisition-to-write reported standard deviation | 1.067152 | 0.799171 | 0.774912 |
| Acquisition-interval reported standard deviation | 0.559981 | 27.797043 | 0.541290 |
| Publication-interval reported standard deviation | 1.703262 | 27.827214 | 1.269744 |

Compared with the earlier optimized capture, the new WMA mean is
**4.756913 us (14.09%) lower**, and its average reported standard deviation
is **0.292240 us (27.39%) lower**. Acquisition-interval spread is back near
the earlier optimized level; publication-interval spread is 0.433518 us
(25.45%) lower in these summaries. No statistically significant improvement
in acquisition jitter is inferred from the small 0.018691 us difference.

The sleep trial's WMA mean is only 0.177506 us (0.62%) above the earlier
continuous-spin capture, while its large acquisition/publication jitter is
absent. This is a cross-capture numerical comparison, not an isolated estimate
of sleep overhead. All ten WMA means fall within 28.968264–29.037761 us and
reported WMA standard deviations within 0.755712–0.801976 us. Acquisition
standard deviations span 0.506091–0.699818 us, and publication standard
deviations span 1.24050–1.33308 us; no windows were discarded.

The result supports the intended experiment: retain most of the two-core
latency benefit while removing the cadence penalty observed with continuous
spinning. It strengthens the CPU-occupancy/RT-budget explanation, but no
scheduler throttle counters or trace were supplied, so the precise mechanism
remains an inference. These values are arithmetic means of rolling-window
summaries, not pooled frame statistics, independent trials, tail bounds, or
complete-loop latency. The comparison references earlier captures rather
than a new adjacent numeric optimized interval.

Next, repeat a settled optimized/parallel/optimized comparison at **3.6 kHz**
with `sleep_us=100`, keeping placement, partition, reader load, instrumentation,
gain and synchronization fixed. Follow with a longer run checking frame
delivery, tails and complete-loop behavior. Retain the generic optimized
default and the explicit parallel option while completing those checks; no
implementation, scheduling budget or instrument configuration changes are
made from this report alone.

Parsed data and calculations are in `/tmp/ocam-sleep100-rtc-20261118.json`.
The engineering plan and operator comparison guide are the only changed
files. Their numerical checks and `git diff --check` passed; no code tests or
instrument commands were run for this documentation-only update.


## Sleep then spin RTC result at 3.6 kHz — 2026-11-18

The operator supplied eight further `fgtimes` reports labeled "3600" as the
continuation of the 100 us sleep-then-spin parallel trial. Log timestamps span
2026-11-19T02:27:52.326837714–02:27:59.334074920; this entry uses the
client-local date 2026-11-18. Average acquisition/publication intervals are
277.832054/277.831924 us; reciprocal mean acquisition interval is about
3,599.3 FPS, consistent with the requested approximately **3.6 kHz** rate.
No new numeric optimized reference at this rate accompanies the excerpt.

Arithmetic averages of every supplied window summary, in microseconds:

| Metric | Earlier 2 kHz parallel, 100 us sleep | 3.6 kHz parallel, 100 us sleep |
| --- | ---: | ---: |
| Acquisition-to-write mean | 29.003462 | 28.861054 |
| Acquisition-to-write reported standard deviation | 0.774912 | 0.975306 |
| Acquisition-interval reported standard deviation | 0.541290 | 0.591119 |
| Publication-interval reported standard deviation | 1.269744 | 1.367250 |

Mean WMA changes by only **-0.142408 us (-0.49%)** relative to the 2 kHz
sleep trial. Its average reported standard deviation increases
**0.200394 us (25.86%)**, so the higher-rate result is not described as
identical jitter. Acquisition/publication standard deviations increase by
0.049829/0.097506 us, respectively; neither approaches the 20–37 us spreads
observed during continuous spinning.

All eight WMA means are 28.737783–28.938933 us; their reported standard
deviations are 0.921077–1.099750 us. Acquisition standard deviations range
0.528989–0.686916 us and publication standard deviations range
1.21748–1.48706 us. All windows are included. These short summaries support
approximately 29 us writer latency with the large cadence regression still
absent at the camera's stated maximum rate. They do not establish the WMA
advantage over optimized at 3.6 kHz: the earlier optimized 33.760 us reference
was collected at 2 kHz and must not be reused as a matched-rate control.

The remaining useful qualification is a longer fixed-condition
optimized/parallel/optimized comparison at the intended operating rate,
checking frame delivery, latency tails and complete-loop behavior. There is
no need to change the 100 us setting based on these summaries alone. The
reported mean/standard-deviation values do not bound rare wake overshoots,
prove absence of frame loss, or provide scheduler-throttling evidence.

Parsed data and calculations are in
`/tmp/ocam-sleep100-3600-rtc-20261118.json`. As with the earlier results, these
are arithmetic averages of rolling-window summaries rather than pooled
frame distributions or independent trials. Only this plan and the operator
comparison guide changed; numerical/whitespace checks passed, and no code
tests or instrument commands were run for this documentation update.

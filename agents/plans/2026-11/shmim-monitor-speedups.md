# shmim reader costs on the AO loop

Branch: `jrmales/shmim-monitor-speedups`

## Problem

`streamCircBuff` was introduced so that one reader of each hot-path stream fans
frames out to the slower readers, keeping the loop's semaphore wake costs
(~1-3 us each, from earlier work) to a minimum. Running `aol1_imWFS2_cbuff`
nevertheless added ~10 us to the loop's end-to-end latency, more than a single
wake should cost.

Questions investigated:

1. Is it the order in which the writer posts the semaphores?
2. Is it a priority mismatch between cbuff and the cacao processes?
3. Is cbuff doing extra work before it is woken?

## Relevant mechanics

- `ImageStreamIO_UpdateIm()` posts every semaphore of the stream in index order
  0..`md->sem`-1, in the writer's context (`ImageStreamIO_sempost(image, -1)`).
  A post to a semaphore with a waiter is a `futex_wake` syscall; without a
  waiter it is only an atomic increment.
- `dev::shmimMonitor` always requests semaphore 5 (`m_semaphoreNumber{ 5 }`,
  not configurable). milk's GPU MVM loop also requests 5. Whichever starts
  second gets the lowest free index.
- In `shmimMonitor::smThreadExec()` nothing happens before the wake except
  `clock_gettime` and `sem_timedwait`, so (3) is ruled out: all cbuff work is
  in its own threads after the wake.
- milk `MVMextractModes` waits for the GPU with `cudaStreamSynchronize` and does
  not call `cudaSetDeviceFlags`. With 128 logical CPUs, CUDA's default
  schedule mode spins, so wfs2cmodeval holds its CPU until it publishes.

## Findings on the RTC (exao2, AMD Threadripper, 8-core L3 domains)

`semreaders.py` (in `shmim-monitor-speedups-files/`) lists a stream's
semaphore readers in post order with each thread's policy, RT priority,
allowed CPUs, cpuset and context switches. Read-only.

Initial state of `aol1_imWFS2`:

| process | CPU | RT prio | sem index |
|---|---|---|---|
| acquWFS (writer) | 45 | 49 | - |
| wfs2cmodeval | 45 | 48 | 0 |
| streamCircBuff sm + fg threads | 26 | 2 | 5 |

1. **Ordering was not the cause**: the MVM was already posted first.
2. **Priority was a cause, but not via cbuff**: acquWFS and wfs2cmodeval share
   CPU 45 with the writer at higher priority. After acquWFS posts sem[0], the
   MVM cannot run until acquWFS finishes the rest of its post loop (including
   the futex wake of cbuff) and blocks. Index order is irrelevant when the
   woken reader cannot run in parallel with the writer.
   - Fix applied in cacao settings by the user: acquWFS -> wfs2cmodeval ->
     mfilt -> mvalC2dm at RT priority 46/47/48/49, so each stage preempts its
     upstream. This made a large improvement; loop latency with no extra
     readers is now ~144 us end-to-end.
3. **cbuff still added ~8 us after the priority change.** Tests:
   - Test A, wake only: cbuff stopped, a semaphore waiter that never reads the
     pixels on CPU 27 (same L3 domain as cbuff):
     `sudo cset proc --set spare27 --exec -- chrt -f 2 shmimInfo -n aol1_imWFS2 -N 2000000`.
     Result: at most ~1 us. The wake is no longer the cost.
   - Test B, cbuff moved from CPU 26 (L3 24-31) to `spare42` (CPU 42, L3
     40-47, same as acquWFS/MVM on 45). Result: ~8 us -> ~2 us.
   - Conclusion: the cost is cache coherence. `aol1_imWFS2` (120x120 float,
     57.6 KB) is rewritten in place every frame. cbuff reading it on another
     CCD leaves ~900 lines shared there, and acquWFS's next write must
     invalidate each one through the I/O die. Within one L3 domain this is
     several times cheaper.

Note: `taskset -c 26 ...` from a shell fails with `EINVAL` because the shell's
cpuset (`system`, 0-23,64-87) does not contain CPU 26; use `cset proc --exec`.

## Tried and reverted: non-temporal output copy

Commit `65cb621a` changed `streamCircBuff::loadImageIntoStream()` to copy into
the output circular buffer with SSE2 non-temporal stores (`_mm_stream_si128`,
then `_mm_sfence` before publishing). The aim was to keep cbuff's writes into
the 40000-frame (2.3 GB) output buffer from evicting the loop's data once cbuff
shares the loop's L3 domain. It passed a unit test against `memcpy` for all
destination alignments.

Result on the RTC with cbuff on `spare42`: the loop got 1-2 us *worse*,
confirmed by several A/B tests. Reverted to `memcpy` in the next commit.

Likely explanation: with regular stores the ~57.6 KB per frame stays in cbuff's
L2/L3 and is written back gradually as it is evicted. Non-temporal stores send
it out of the CCD as one ~900-line burst right after each frame, over the CCD's
link to the I/O die. That link is shared with acquWFS, the MVM and the DM
threads on CPUs 40-47, and moving 57.6 KB at tens of GB/s takes ~1-2 us. It
also shows the L3 churn from the output buffer was not a significant cost.

Lesson: on this machine, avoid bursty writes out of the loop's CCD from
co-located helpers.

## camwfs-avg (shmimIntegrator) on the camera stream

With cbuff on `spare42`, running `camwfs-avg` added another ~8-10 us. It reads
`camwfs` directly, which ocam2KCtrl (FIFO 90, CPU 33, L3 32-39) writes as a
single buffer (`circBuffLength=1`, 120x120 uint16 in bin2) that acquWFS (CPU
45) reads. camwfs-sw and camwfs-dark were off for all of these tests.

- Semaphores of `camwfs`: acquWFS is sem[0]; camwfs-avg sem[5], unpinned FIFO 2
  in the `system` cpuset. Here the writer and acquWFS are on different CPUs, so
  acquWFS runs as soon as it is posted and later wakes are off its path.
- Wake only (unpinned `chrt -f 2 shmimInfo -n camwfs`, sem[9]): 1-2 us or less.
  Not being on a cpuset is not the cause.
- camwfs-avg moved onto CPU 32 (the writer's L3 domain, `cpuset=camwfs_sw`
  for its monitor and framegrabber): no change. Not cache traffic between
  CCDs, unlike cbuff.
- `camwfs-avg.start` off (wake still happens, `processImage` returns before
  reading the pixels): loop back to ~144 us. The cost is the per-frame pixel
  work.

The per-frame work converted each pixel through a function pointer
(`getPix` via `m_pixget`), which cannot be inlined or vectorized, reading the
camera buffer for the whole conversion. Benchmark on a workstation, 120x120
uint16 with opaque function pointers as in the app: ~17.5 us per frame.

### Change on this branch

- `getPixFrame` / `getPixFramePointer` in `libMagAOX/ImageStreamIO/pixaccess.hpp`:
  convert a whole frame of a known type, setting or adding to the destination,
  in a loop the compiler can vectorize. Selected once per allocation, like
  `getPixPointer`.
- `shmimIntegrator` copies each source frame into a private buffer with one
  `memcpy` (`m_staged`), then converts from the copy with the frame converters,
  for the source, moving-average history and darks. The source buffer is now
  read for well under a microsecond instead of ~17 us.
- Benchmark after the change: ~1.75 us per frame including the copy.

Verification: `ImageStreamIO_test` (replacing an empty template) checks the
frame converters against `getPix` for all ten types, setting and adding, with
no writes past the frame (782 assertions); `shmimIntegrator_test` 20 cases,
2003 assertions pass; app builds.

Result on the RTC (new build confirmed running): **no change** in the loop
cost of camwfs-avg; the rms of the end-to-end measurement also did not change.
The integrator change is kept for its ~10x lower CPU use in every instance.

Further test: camwfs-sw alone (streamWriter, not writing to disk, so only a
`memcpy` of each frame into its buffer) on CPU 32 shows the same ~8 us.

Summary for `camwfs`:

| reader | loop cost |
|---|---|
| wakes, does not read (shmimInfo, or camwfs-avg with `start` off) | ~1-2 us |
| reads the frame: camwfs-avg slow or fast, CPU 0-23 or 32; camwfs-sw | ~8-10 us |

Any read of the frame costs a fixed ~8 us, independent of where the reader
runs and how long it holds the frame. Working hypothesis: the semaphore wakes
every reader at the same moment as acquWFS, so they read the same lines (owned
by the camera core, 33) at the same time and acquWFS's misses queue behind the
other reader's. Not yet confirmed. ~8 us is also the size of the MVM
completion-wait jumps removed in xwcl/milk `dc0aea87`, so a threshold effect in
the loop remains possible.

### shmimMonitor spin polling and read delay

Two `dev::shmimMonitor` options (config section of each monitor, e.g.
`[shmimMonitor]`):

- `spin` (bool, default false): poll `md->cnt0` (atomic acquire load, with
  `_mm_pause`) instead of waiting on the semaphore. Shutdown, restart and state
  are checked every 1024 polls; the 1 s timeout and file/inode checks are
  unchanged. A semaphore is still registered (so the release logic is
  unchanged); with no waiter the writer's post is only an atomic increment.
  Uses a full core: a warning is logged if no cpuset is set.
- `readDelay` (int, microseconds, default 0): sleep (`clock_nanosleep`) after a
  new frame is seen, before reading it. Tests the collision hypothesis and, if
  confirmed, is a cheap fix for a single-frame source with a long frame period.

Tests: `shmimMonitor_timing_test` gains a polling case (frames seen from the
counter alone with no semaphore post, and clean shutdown) and a read-delay case
(200 ms delay respected in both modes); 8 cases, 109 assertions, 5 repeated runs
pass. streamCircBuff and shmimIntegrator tests and apps build and pass.

New config keys must be deployed with this code (unknown keys fail startup).

Results on the RTC, camwfs-avg reading `camwfs`:

| mode | loop cost |
|---|---|
| semaphore, `readDelay` 30-50 us | ~8 us, unchanged |
| `spin=true`, dedicated cpuset (`camwfs_sw`, CPU 32) | none |
| `spin=true`, `system` cpuset | ~8 us |

The simultaneous-read hypothesis is ruled out. A reader on a dedicated core
that never sleeps costs nothing. Unresolved: whether the cost comes from the
reader's core going idle and waking (C-state exit) or from where it reads, since
the two runs that cost 8 us differ from the free run in both: the semaphore
reader on CPU 32 sleeps between frames, and the system spinner reads from
another CCD (0-23) and shares its core with SMT siblings, IRQs and
housekeeping. Earlier, cbuff on `aol1_imWFS2` with the semaphore cost ~2 us on
CPU 42 but ~8 us on CPU 26, which fits location better than sleeping.

Follow-up results:

- Semaphore reader on CPU 32 with deep C-states disabled for CPU 32: still
  ~8 us. In RT mode sysMonitor `set_latency` already sets the performance
  governor on all CPUs and holds `/dev/cpu_dma_latency` at 0, so no CPU uses
  deep C-states. C-state exit is ruled out.
- `spin=true` on `spare27` (CPU 27, a different CCD from the camera on 33):
  free, loop at 144 us. Cross-CCD reading is ruled out.

- `readDelay` had already been tried at 100 us as well as 30-50 us: no
  effect. At 100 us acquWFS's calibration (~67 us) is over, so a collision
  with acquWFS's own read of the frame is unlikely.

**Correction:** camwfs-avg produced no output frames in spin mode, so the
"free" spin results are invalid. Likely cause: its monitor and framegrabber
threads were both on the same dedicated cpuset (as in the CPU 32 test), both
SCHED_FIFO at the default priority 2. A spinning FIFO thread never blocks, so
the framegrabber on that core never runs and nothing is published. With the
completed average never claimed, `processImage` then returns before reading
the pixels (`m_updated` stays set), so camwfs-avg stopped reading `camwfs`.
Retest with the monitor thread alone on its core, e.g. `[shmimMonitor]`
`cpuset=spare27`, `spin=true`, and `[framegrabber]` on `system`.

Valid retest: camwfs-avg monitor spinning (`spin=true`, `spinSleep=0`,
`threadPrio=2`) with its framegrabber at `threadPrio=3` on the same cpuset
(`camwfs_sw`, CPU 32), `camwfs_avg` updating: the ~10 us increase is back.
So spinning does not help either: **any reader that reads the `camwfs` pixels
costs ~8-10 us**, however it is woken (semaphore or polling), wherever it runs
(CPU 0-23, 27 or 32), with idle states off, and with the read delayed up to
100 us. A reader that never reads the pixels costs ~1-2 us.

With the read delayed 100 us, acquWFS's read and calibration are already over,
so the cost most likely lands on the camera's next write of the single-buffer
`camwfs` (ocam2KCtrl descramble on CPU 33, `circBuffLength=1`), which must take
back every line the extra reader holds. This is the same mechanism as cbuff on
`aol1_imWFS2`, where a reader in the writer's L3 domain was cheap (~2 us on 42
vs ~8 us on 26), but here a reader in the camera's L3 domain (32) still costs
~8-10 us. Unexplained so far. Note that every location tested for `camwfs` is
outside acquWFS's L3 domain (40-47), whereas the cheap cbuff placement was in
it.

Result: the camera's own acquisition-to-write time (camwfs `fg_timing`
`delta_aw`, the "wma" time) rises by the same ~8-10 us when a reader reads the
pixels. The cost is in the camera's write.

ocam2KCtrl's `ocam2_descramble` writes directly into the `camwfs` buffer.
The vendored SDK (`apps/ocam2KCtrl/fli/ocam2_sdk.c`,
`ocam2_descramble_binning2x2`) shows it is a gather: scattered reads from the
raw EDT DMA buffer through `g_descrblTab`, and two sequential write streams
into the image (`image[i]`, `image[chunk+i]`). (An earlier guess that its
stores were scattered was wrong.) The gather is bounded by its read misses (the
code notes a plain `memcpy` would be ~10 us faster). When a reader holds the
stream's lines from the previous frame, each store also needs its line taken
back, and those requests compete with the gather's misses for the core's
miss-handling resources. This is the working hypothesis, not confirmed.

Change: `camera.stagedDescramble` (bool, default false). When set and digital
binning is off, `loadImageIntoStream` descrambles into a private buffer
(`m_descrambled`, sized on first use), which stays in the camera core's cache,
so the gather has no line takebacks to compete with, and then copies it to the
stream with one sequential `memcpy`, which takes the lines back on its own. Digital binning already descrambles into
`m_digitalBinWork`. Test: a new `ocam2KCtrl_test` section checks the
descramble writes the private buffer and the destination receives the frame;
all ocam2KCtrl suites pass. The app needs the EDT SDK and was not built on the
workstation.

**Result on the RTC: `stagedDescramble=true` removes the ~8-10 us cost of
camwfs-avg reading `camwfs`.** This confirms the cost was the camera's
descramble being slowed by line takebacks from readers holding the previous
frame.

With `stagedDescramble=true`, camwfs-avg and camwfs-sw reading `camwfs`
directly cost essentially nothing. Stopping acquWFS lowers the camera's write
time (wma) by ~3 us: acquWFS, the loop's own reader on CPU 45 in the other L3
domain, still holds the lines the sequential copy has to take back each frame.

General lesson: a writer that fills a shared single-buffer stream while also
taking cache misses (a gather, or a computation over other data) slows down
when readers hold the stream's lines from the previous frame. Writing into a
private buffer and then copying it to the stream sequentially avoids this.
cacao acquWFS writing `aol1_imWFS2` while it calibrates is the same pattern,
which may explain the cbuff cost seen there (~8 us from CPU 26, ~2 us from 42).

Originally planned measurement: `delta_aw` and the loop time with camwfs-avg reading,
`stagedDescramble` on vs off. It may also lower the baseline, since acquWFS
already holds the frame's lines from the previous frame.

Earlier next measurements:
1. camwfs `fg_timing.delta_aw` (mean acquisition-to-write time, includes the
   descramble) with camwfs-avg reading vs not. If it rises ~8-10 us, the cost
   is in the camera's write.
2. A `camwfs` reader in acquWFS's L3 domain (40-47), by analogy with cbuff on
   42.

Earlier summary, before the valid retest: a reader that reads `camwfs` costs ~8 us when it is woken by the
semaphore, whatever its CPU, CCD, idle states or read delay (0-100 us), and
nothing when it spins on a dedicated core in either CCD. A semaphore waiter
that does not read costs ~1-2 us. The mechanism is not identified. Localizing
it would need per-process timing with the semaphore reader on vs. spinning,
e.g. acquWFS, wfs2cmodeval, mfilt and mvalC2dm in `milk-procCTRL`, and camwfs's
own acquisition-to-write time.

Superseded by the valid retest: a `camwfs` reader spinning on a dedicated core is
**not** free. The earlier placement outcome was: free, in the camera's L3 domain or another. With the planned `camwfs` cbuff, that is one
dedicated core running cbuff with `spin=true`, and camwfs-avg, -sw and -dark
reading its output.

Placement plan (user): a `camwfs` cbuff in the writer's L3 domain, with
camwfs-avg, camwfs-sw and camwfs-dark reading its output instead of `camwfs`.
The integrator change still helps every shmimIntegrator instance (all
`*-avg`/`*-dark` apps, which use block averaging).

## To do on this branch

1. Done: spin polling in `dev::shmimMonitor` (`spin`, `spinSleep`,
   `readDelay`) and `streamWriter` (`framegrabber.spin`,
   `framegrabber.spinSleep`), sharing `MagAOX::app::pollFrame`
   (`libMagAOX/app/framePoll.hpp`).
   - A frame is new when `md->cnt0` changed and `md->write` is 0. frameGrabber
     sets `cnt0` before the per-frame arrays and clears `write` last, so `cnt0`
     alone could be seen before `cntarray` is updated.
   - `spinSleep` (us) sleeps between polls instead of pausing, freeing the core
     at the cost of up to that latency. Without it a cpuset is needed (a
     warning is logged otherwise).
   - The spinning thread must be alone on its core: a same-priority FIFO
     thread on that core (e.g. the app's own framegrabber) never runs.
   - A semaphore is still registered in spin mode.
   - Tests: `framePoll_test` (fully published frame only, timeout, stop, with
     and without sleep); `shmimMonitor_timing_test` polling case for
     `spinSleep` 0 and 100; `streamWriter_lifecycle_test` real workers ingest
     frames published without semaphore posts, `spinSleep` 0 and 100. All
     streamWriter, streamCircBuff and shmimIntegrator tests pass; apps build.
2. Done: spin retest with the framegrabber at higher priority. Spinning does
   not remove the cost (see above).

## Follow-ups (not on this branch)

- Loop time with `stagedDescramble=true`: ~146 us with camwfs-avg and
  camwfs-sw off (vs 144 us unstaged), and <1 us more with them on. Staging
  costs ~2 us when acquWFS is the only reader (its copy takes back acquWFS's
  lines), and saves ~8-10 us per extra reader.
- Added `camera.stagedCopyNT` (bool, default false): with `stagedDescramble`,
  copy to the stream with `MagAOX::streamCopy` (`libMagAOX/common/streamCopy.hpp`,
  SSE2 non-temporal stores and a store fence; the copy from the reverted
  streamCircBuff commit, now shared, with `streamCopy_test`). To A/B on the
  RTC against plain `stagedDescramble`.
- **Result with `stagedDescramble=true` and `stagedCopyNT=true`:** the
  camera's write time (wma) is a steady ~37 us whether acquWFS, camwfs-sw and
  camwfs-avg are reading or not, and the loop is 142-143 us (below the original
  144 us baseline) independent of camwfs-sw and camwfs-avg, with camwfs-avg
  unpinned in the `system` cpuset. The burst risk seen with cbuff's
  non-temporal copy did not appear here: the frame must leave the camera's CCD
  for acquWFS anyway, and the camera no longer waits to take lines back.
- Superseded by the result above: the remaining ~3 us of camera write time from acquWFS: candidates were
  writing the staged copy to the stream with non-temporal stores (no takebacks,
  but readers then read from DRAM, and the earlier cbuff experiment showed
  bursts of non-temporal stores can hurt), or placing the camera thread and
  acquWFS in the same L3 domain.
- With direct readers now free, the `camwfs` cbuff is unnecessary.
- Moved to `dev::frameGrabber` (user decision; camwfs is the only OCAM camera):
  `framegrabber.staged` (bool, default false) has the base pass
  `loadImageIntoStream` a private buffer (`m_stagedImage`, sized at acquisition
  start), set `md->write` and copy the buffer into the stream slot;
  `framegrabber.stagedCopyNT` (bool, default true) makes that copy
  `MagAOX::streamCopy`. Any camera can opt
  in; derived classes, flips and descrambles are unchanged since they just write
  to `dest`. The ocam2KCtrl options (`camera.stagedDescramble`,
  `camera.stagedCopyNT`) are removed and ocam2KCtrl is back to its original
  `loadImageIntoStream`. Test: a `frameGrabber_integration_test` case runs the
  real loop unstaged, staged with `memcpy` and staged with `streamCopy`, checking
  where the conversion wrote and the published pixels and counters.
- RTC result with the first frameGrabber version: works, but camwfs-sw and
  camwfs-avg added a small, repeatable ~2 us, which the ocam2KCtrl version did
  not. The one difference found: that version set `md->write = 1` just before
  the copy rather than before the load, so taking back the metadata line (read
  by every reader each frame) moved from overlapping the load to directly before
  the copy and its closing store fence. Restored `write = 1` before the load in
  both modes, matching the ocam2KCtrl version. RTC result: fixed, wma back to
  ~37 us and the loop to ~143 us with camwfs-sw and camwfs-avg running.
- Deployment: set `[framegrabber] staged=true` in `camwfs.conf`, and remove
  `camera.stagedDescramble` / `camera.stagedCopyNT` if they were added for
  testing (unknown keys fail startup).
- Done differently 2026-11-14: acquWFS copies `aol1_imWFS2` to
  `aol1_imWFS2pub` for readers outside the loop (see the last section).

- shmimIntegrator moving-average mode (`nUpdate` > 0, not used in any deployed
  config) re-sums the whole history on every update, O(nAverage x pixels).
  A running sum would make it O(pixels).
- `lowfs-refsub-avg` reads `aol2_imWFS2` directly; check whether that loop
  has the same exposure.
- `libMagAOX/ImageStreamIO/ImageStruct.hpp` uses `std::numeric_limits`
  without including `<limits>`; it compiles only because of include order.

- Permanent core for `aol1_imWFS2_cbuff` in the 40-47 L3 domain:
  `rtSetup/RTC/make_cpusets` reserves `spare42` to keep the BMC SDK worker out
  of the MVM's domain, and 40-47 has no other free core. Needs a decision.
- `aol1_modevalWFS_cbuff` (CPU 50, L3 48-55) reads mfilt's input (mfilt on 46)
  across CCDs. Measured 2026-11-14: stopping it lowers the loop by ~3 us. It
  needs a core in 40-47 along with `aol1_imWFS2_cbuff`, and `spare42` is the
  only free one.
- Resolved 2026-11-14: ImageStreamIO large-stream fix. Neither `xwcl/milk`
  branch had ever pinned the fix (`57cef04`, ImageStreamIO PR #70); it had been
  checked out by hand on the RTC, and setup's `git submodule update` reset it.
  xwcl/milk PR #5 merged `magaox` into `dev` and pinned ImageStreamIO `dev`
  (`3b36650`); `magaox` was then fast-forwarded to `dev` in both xwcl/milk
  (`7b7d99e9`) and jaredmales/cacao (`ced2187d`), so `dev` and `magaox` are
  identical. Milk `dev` does not build against the old cacao `magaox`
  (`compute_SVD` API). The RTC already runs milk and cacao `dev`; it needs
  `git pull && git submodule update` and a milk rebuild. Its installed cacao
  libraries (2026-11-01) may predate the `.comp.WFSrefsubNT` default changing
  from 1 to 0, so check that setting after the rebuild.
- Make the shmimMonitor semaphore index configurable only if a case appears
  where post order matters again (readers able to run in parallel with the
  writer).

## Loop time check after restart (2026-11-14, paused)

After a restart the loop measured 145 us against ~137 us the day before. The
camera side was unchanged (camwfs wma 37-38 us), RT mode (`set_latency`) was on,
and the cacao priorities were 46/47/48/49. With `aol1_imWFS2_cbuff`,
camwfs-avg and camwfs-sw stopped, `camwfs` and `aol1_imWFS2` each had one
reader. The MVM stages were 29 and 20 us.

| Change | Loop time |
|---|---|
| Starting point: extra readers off, synco enabled | 145 us |
| Stop `aol1_modevalWFS_cbuff` | 142 us |
| Disable synco (process left running) | 140 us |

- The ~137 us runs had synco disabled, so 140 us is the comparable number;
  enabling synco costs ~2 us.
- Stopping the synco process stops `holoop-time` from updating. This is by
  design: with automatic synco integration the loop timing monitor needs synco's
  main-command labels in both modes (`loop_timing_monitor.md`).
- A full-thread listing showed every RT thread on its planned core
  (`agents/plans/2026-10/rtc-cpuset-restructuring.md`): bmcCtrl command thread
  on 40, post worker on 44 and SDK worker on 56; dmSyncopation on 41 and 57;
  2x dmSpeckle (tweeterSpeck, tweeterMod) FIFO60 on 47; camwfs framegrabber
  on 33. Nothing else was on 40-47 apart from per-CPU kernel threads.
- The installed cacao (2026-11-01) and cacao `dev` both contain the acquWFS
  vectorization and `.comp.WFSrefsubNT`, so the build is not the cause.
- **Open:** ~3 us is unexplained. Placement does not show activity; the
  modulators on CPU 47 (FIFO60, in the loop's L3) are untested. Next time,
  record per-stage timings (acquWFS stages, MVMs, dmtweeter SDK) at the 140 us
  baseline so a future shift can be traced to a stage.

## acquWFS publish copy for readers outside the loop (2026-11-14)

With synco disabled (140 us baseline), turning on `.comp.WFSrefsubNT` with
`aol1_imWFS2_cbuff` running made the loop up to 5 us worse. Non-temporal
`imWFS2` stores push the frame to DRAM, so the same-core reader
(wfs2cmodeval's gather on 45) loses it from cache.

Instead, cacao `jrmales/acquwfs-pubcopy` (jaredmales/cacao `4721359d`) adds two
hidden acquWFS FPS settings, both default OFF:

- `.comp.WFSpub`: after publishing `aol1_imWFS2`, copy it to
  `aol1_imWFS2pub` and publish that with the same `atime`. The MVM is on the
  same CPU at higher priority, so the copy runs only while the MVM is not using
  it. Only CPU 45 and the GPU touch `imWFS2`.
- `.comp.WFSpubNT`: make that copy with non-temporal AVX stores.

Deployment: recreate the acquWFS-1 FPS (new parameters), turn on
`.comp.WFSpub`, and set `shmimName=aol1_imWFS2pub` in
`aol1_imWFS2_cbuff.conf` (a value change only). acquWFS must start before
readers of `aol1_imWFS2pub`.

**RTC result:** with `.comp.WFSpub` on and the cbuff reading `aol1_imWFS2pub`,
the loop time does not change when `aol1_imWFS2_cbuff` is on. The cbuff no
longer needs a core in the loop's L3 domain.

Next:
- Move the other direct readers of `aol1_imWFS2` to `aol1_imWFS2pub`:
  `dlDataCollection`, `nnReconstructor`, `sparkOG`, `dlReconstructor`,
  `isRTC` and the mzmq calib server/client.
- The same pattern for `aol1_modevalWFS` (written by wfs2cmodeval in milk's
  MVM), so `aol1_modevalWFS_cbuff` (~3 us) can also stay off the loop's L3.

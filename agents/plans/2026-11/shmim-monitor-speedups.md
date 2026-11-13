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

To measure on the RTC: loop time with camwfs-avg running, against ~144 us.

Placement plan (user): a `camwfs` cbuff in the writer's L3 domain, with
camwfs-avg, camwfs-sw and camwfs-dark reading its output instead of `camwfs`.
The integrator change still helps every shmimIntegrator instance (all
`*-avg`/`*-dark` apps, which use block averaging).

## To do on this branch

1. Spin polling in `dev::shmimMonitor`: optional mode that polls `md->cnt0`
   (and `cnt1`) instead of waiting on the semaphore, for readers with a
   dedicated core. The writer's post then has no waiter and no syscall.
   Design points:
   - Configurable, default stays semaphore wait.
   - Do not register a semaphore in spin mode.
   - Use `_mm_pause` in the poll loop; the metadata line still moves between
     cores once per frame.
   - Keep the shutdown, restart, size-change and file/inode checks, run on a
     time interval instead of on semaphore timeout.
   - Single-frame sources can be overwritten before they are read; keep cbuff's
     torn-frame check.
   - Test A suggests the gain per reader is at most ~1 us on the current
     configuration.
2. Spin polling in `streamWriter`, which has its own wait loop
   (`sem_timedwait` then drain with `sem_trywait`, `streamWriter.hpp` ~1541).
   Same design points.

## Follow-ups (not on this branch)

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
  across CCDs; likely the same cost. Repeat tests A and B on it.
- ImageStreamIO large-stream fix regression: `xwcl/milk` `magaox` still pins
  ImageStreamIO `21a8ae1`; the fix (`57cef04`, ImageStreamIO PR #70, based on
  `21a8ae1`) was never pinned, so opening streams over 2 GB fails
  ("Error in the file (too small)"). Bump the submodule pointer; it is a
  fast-forward touching only the size checks.
- Make the shmimMonitor semaphore index configurable only if a case appears
  where post order matters again (readers able to run in parallel with the
  writer).

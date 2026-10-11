# Task Description
Create a plan to investigate why the Hamamatsu image stream is experiencing high latency. 

## Problem Statement
I should be able to stream images at 2kHz in a 128x128 ROI but it's instead capped at ~780 Hz despite the exposure time being set to its minimum value (7.2e-06s).

## Discussion
- The 780 Hz FPS that I am referencing corresponds to `milk-shmimmon camham` which is the rate at which images are being streamed out from the camera to the computer. 
- The issue might be appearing within the `DCAMWAIT functions` contained in `/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/dcamapi4.h`. It's possible that the camera is waiting until the first frame has been written out before reading in the second frame which could be causing the delay. 
- Consult the camera manual `ham_doc.md` as needed.Pages 72 and 73 of the manual contain relevant information on the camera's timing settings.
- The Orca Quest 2 supports a `frame bundling` mode which, in theory, allows the camera to run up to `19841 Hz` with 4 vertical lines in the camera's `Standard Scan` mode. Consider this the stretch goal.

## Requirements
1. When I set ROI to 256x256 with the shortest exposure time, the FPS reported in `milk-shmimmon camham` should report as approx. 1030 +/- 10 Hz.
2. When I set ROI to 128x128 with the shortest exposure time, the FPS reported in `milk-shmimmon camham` should report as approx. 2000 +/- 10 Hz.
3. The FPS reported by the `camham.fps` INDI prop should closely match that which is reported by `milk-shmimmon camham`


## Tests and Metrics
- Test that I can seamlessly switch between different ROIs
- Test that the FPS reported by `milk-shmimmon camham` updates when I switch to a new ROI.
- Test that the value reported by `camham.fps` matches that which is reported by `milk-shmimmon camham` to within 10 Hz.

## Scope and Caveats
None.
# Instructions to Agent

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

## Task Summary
<!-- The agent should summarize the task as they understand it -->

The `camham` stream (orcaCtrl, ORCA-Quest2 over CoaXPress) should publish every frame the camera
produces: about 2000 Hz at 128×128 and about 1030-1040 Hz at 256×256 with the minimum exposure. The INDI
`camham.fps` value should agree with what `milk-shmimmon camham` measures. Today the stream runs at about
760-780 Hz at 128×128 even though the camera itself runs at about 2013 Hz.

### Findings (2026-10-05, read-only: live INDI, `/proc`, DCAM headers, manual)

Live values with the camera OPERATING at 128×128 ROI, Standard scan:

| INDI value | Reading | Meaning |
| --- | --- | --- |
| `camham.exptime.current` | 7.2e-6 s | minimum exposure, so exposure is not the limit |
| `camham.readout_time.value` | 0.000482 s | about 67 lines × 7.2 µs |
| `camham.fps.current` | **2012.9 Hz** | DCAM's `INTERNALFRAMERATE`; matches the manual (`1 / ((128/2 + 5) × 7.2 µs)` = 2013 fps) |
| `camham.fg_timing.acq_fps` / `write_fps` | **758 Hz** | the rate frames are actually published, which matches shmimmon |
| `camham.fg_timing.acq_min` / `acq_max` | **0.80 ms** / 3.2 ms | even the *fastest* loop iteration spans about 1.6 camera frames |
| `camham.fg_timing.delta_aw` | 15 µs | copying and publishing a frame is cheap |

1. **The camera isn't the bottleneck.** It runs free (`DCAMCAP_START_SEQUENCE`, internal trigger) at
   about 2013 fps into a DCAM ring buffer and doesn't wait for the host. On the Discussion's
   hypothesis: in sequence capture the camera never waits for the host to read a frame, so it isn't
   "waiting for the first frame to be written out". The host is falling behind, and the ring buffer
   absorbs that.
2. **The host drops frames by design.** `acquireAndCheckValid()` calls `dcamwait_start( FRAMEREADY )`,
   then `dcamcap_transferinfo()`, then locks **only `nNewestFrameIndex`**. Every frame that arrived since
   the last wake-up is skipped. The published rate therefore equals how often the wait returns, not the
   camera rate. The existing "Skipped frame(s)!" console messages (expected 0.497 ms, got about 1.3 ms)
   show this.
3. **The wait returns at most every 0.8 ms (`acq_min`).** The frame grabber thread (FIFO priority 2)
   uses only about 5% CPU, so it is asleep in `dcamwait_start`, not busy. A second orcaCtrl thread
   (TID 2915338, created at capture start, so most likely DCAM/driver's transfer and event thread) uses
   about 9% CPU at **normal (non-real-time) priority**. Event-delivery latency, or event coalescing, in
   DCAM, its thread or the CoaXPress driver limits wake-ups to about 760-1250 per second.
4. `DCAMCAP_TRANSFERINFO.nFrameCount` gives the total number of frames transferred, and
   `DCAMBUF_FRAME.framestamp` numbers each frame. Together they let orcaCtrl publish **every** frame
   transferred since the last wake-up (drain the backlog) instead of only the newest. Each wake-up would
   then publish about 2.6 frames, so the stream matches the camera rate even if wake-ups stay slow.
5. The DCAM ring buffer is only `m_frameCount = 10` frames, which is 5 ms at 2 kHz. That's too small to
   absorb scheduling hiccups once the backlog is drained (`acq_max` is already 3.2 ms).
6. **Frame bundle** (`DCAM_IDPROP_FRAMEBUNDLE_MODE`/`_NUMBER`, plus `_ROWBYTES`/`_FRAMESTEPBYTES`)
   packs N images into one DCAM frame, so N images arrive per event. The manual's 19,841 fps at 4 lines
   requires it. This is the stretch goal and the natural extension of finding 4.
7. Expected rates from the manual (section 11-1-4, free running, Standard scan, CoaXPress, minimum
   exposure): 128 lines gives 2013 fps; **256 lines gives 1 / (133 × 7.2 µs) = 1044 fps** (the table
   rounds this to 1040); 4 lines gives about 19,800 fps with frame bundle.
8. Other things checked and ruled out:
   - CPU contention: 64 cores; the only higher-priority user threads are `DMch2disp-00` (FIFO 49) and
     `dmSpeckle` (FIFO 60).
   - Readout speed is Standard.
   - The exposure-units bugs are fixed (`ab34d998`), and `readout_time` now shows seconds.
   - `circBuffLength=1` doesn't limit the writer, and shmimmon counts `cnt0`.

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

- "FPS reported by `milk-shmimmon camham`" is the rate `cnt0` advances, i.e. the published frame rate.
- The goal is **throughput**, every camera frame published, more than the lowest per-frame latency.
  Draining a backlog publishes some frames up to one wake-up interval (about 1 ms) after they arrive.
  See Q2.
- `camham.fps` should keep meaning the camera's frame rate (`DCAM_IDPROP_INTERNALFRAMERATE`), as
  stdCamera intends. Requirement 3 is met by making the stream keep up, not by changing what `fps`
  reports. See Q3.
- DCAM ring-buffer frame `k` (0-based, counting from capture start) is at index `k % m_frameCount`, and
  `framestamp` increases by 1 per frame. Test T0 checks this before the drain logic relies on it.
- Tests run on the live `camham` while other AO processes are running. Restarting orcaCtrl briefly
  interrupts `camham-avg` and `camham-dark`. See Q4.
- AGENTS.md applies: functional commits first, clang-format, a GCC 14 build, short commit messages,
  app-header-only code, unit tests under `apps/orcaCtrl/tests` (rule 20), and this document kept up to
  date.

## Requirements
<!-- The agent should list the requirements to which they are planning -->

- **R1:** 256×256, minimum exposure: shmimmon reports **1040 ± 10 Hz** (camera calculates 1044; set by
  the Q1 answer).
- **R2:** 128×128, minimum exposure: shmimmon reports **2000 ± 10 Hz** (camera calculates 2013). It was
  briefly revised to about 1545 Hz (Q7), then restored when round 2c showed the 1545 ceiling came from
  attached buffers.
- **R3 note:** below about 128 lines, DCAM delivers fewer frames than the camera's calculated rate
  without frame bundling (round 2c: 64 lines 2885 of 3754 Hz; 32 lines 2320 of 6614 Hz). There,
  `camham.fps` (calculated) can't match shmimmon. R3 is checked at 128 lines and above.
- **R3:** `camham.fps` agrees with shmimmon to within 10 Hz at each ROI.
- **R4:** ROI changes are seamless. The stream restarts at the new rate and `camham.fps` updates, with
  no app ERROR and no manual restart.
- **R5 (derived):** no frames are silently dropped in steady state. Any ring-buffer overrun is counted
  and logged, rate-limited, rather than printed per frame.
- **R6 (derived):** no regression at full frame (about 120 fps) or with binning, and the DCAM buffer's
  memory stays bounded at full frame.
- **Goal for this plan (from the Q6 answer):** run as fast as the manual allows *without* frame
  bundling. Frame bundling (up to about 19,800 fps at 4 lines) moves to a separate plan.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

1. **R1 target:** the manual's free-running formula gives **1044 fps** at 256 lines (the table says
   1040). 1030 is the *external edge-trigger* figure. Should R1 be **1040 ± 10 Hz**, or is the camera
   meant to be externally triggered?
2. **Throughput or latency:** publishing every frame means that after a slow wake-up, 2-3 frames are
   published back to back, the oldest up to about 1 ms late. Is that acceptable for the camham
   consumers? Alternatively, a config option (e.g. `camera.publishAllFrames`, default `true`) could keep
   the current "newest only" behavior for latency-critical use.
3. **What `camham.fps` should report:** keep the camera-calculated rate (stdCamera convention, and it
   updates immediately on an ROI change), or report the measured publish rate? The plan keeps the
   camera rate and verifies it against shmimmon.
4. **Test window:** testing needs orcaCtrl restarts and ROI changes on the live `camham` while the cacao
   loop and the `camham-avg`/`camham-dark` integrators run. When is that OK, and should I run the
   tests or will you?
5. **Privileges:** experiment E3 (raising the DCAM event thread to real-time priority with `chrt`)
   needs to run as `xsup` or root. OK to try, or skip?
6. **Stretch goal timing:** do frame bundle in this task, after R1-R5 pass, or as a separate plan?

### Answers (user, 2026-10-05)

1. R1 is **1040 ± 10 Hz**.
2. **Publish every frame.** No option to keep "newest frame only".
3. Keep the **camera's calculated rate** for `camham.fps`.
4. Restarting orcaCtrl is OK for tests, but **tell the user just beforehand and wait for confirmation**
   each time.
5. **Hold off on E3** (DCAM thread priority). See how far the other steps get first.
6. **Frame bundling is a separate plan.** The goal for now is the fastest rate the manual allows
   without frame bundling.

The old plan, `agents/plans/2026-10/orcaCtrl-latency.md`, was deleted at the user's request; this
document replaces it.

### New question (2026-10-11)

7. **R2 without frame bundling:** round 2b measured a fixed DCAM/driver delivery ceiling of about 1545
   frames/s, independent of ROI height. Is about 1545 Hz at 128×128 acceptable for this plan, with
   2000 Hz moved to the frame-bundle plan? Or should E3 (DCAM thread priority) be tried first?

   **Answer (user, 2026-10-11): accept about 1545 Hz at 128×128.** R2 is revised below. 2000 Hz at
   128×128 moves to the frame-bundle plan, and E3 stays on hold.

   **Correction (2026-10-11, round 2c):** the 1545 frames/s "ceiling" was caused by the attached
   (`dcambuf_attach()`) buffers, not by DCAM or the driver. With DCAM-allocated buffers and the per-slot
   pointer cache, 128×128 streams at **2012.9 Hz**. R2 is restored to its original **2000 ± 10 Hz**,
   and the Q7 premise no longer applies.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

**Unit tests (no hardware; `apps/orcaCtrl/tests/orcaCtrl_test.cpp`, AGENTS.md rules 20-21):**

- **U1, frame bookkeeping:** the drain logic will be a small pure helper (see Implementation step 2).
  Given (next frame to publish, total transferred, ring size), it returns which frame to lock, its ring
  index, and how many frames were lost to overrun. Cases: no backlog; a backlog of 1-3; exactly a full
  ring; overrun by 1 and by many; and the reset on reconfigure. Purpose: make sure the ring indexing and
  overrun recovery can't silently misorder or drop frames.
- **U2, buffer sizing:** the helper that picks the DCAM buffer frame count from the frame size and a
  byte budget (Implementation step 3). Check full frame (18.9 MB/frame), 256×256 and 128×128 against the
  minimum and maximum counts.

**Hardware tests (camham, minimum exposure, Standard scan; results recorded below):**

- **T0, baseline and DCAM indexing** (before changes, instrumented scratch build): log, per wake-up,
  the change in `nFrameCount`, `nNewestFrameIndex` and `framestamp`, and the duration of
  `dcamwait_start`, `transferinfo` and `lockframe`. Purpose: confirm findings 2-4 (frames per wake-up
  about 2.6 at 128×128) and the `k % m_frameCount` index mapping.
- **T1 (R1):** 256×256: shmimmon and `fg_timing.write_fps` over 60 s are within the R1 window, and
  `camham.fps` is within 10 Hz of shmimmon.
- **T2 (R2):** 128×128: shmimmon is 2000 ± 10 Hz, and `camham.fps` is within 10 Hz.
- **T3 (R4):** cycle the ROI 128 → 256 → 1024 → full → 128 through INDI, five times. Each time, the
  stream resumes within about 2 s, shmimmon settles to the new expected rate, `camham.fps` updates, and
  the app never enters ERROR.
- **T4 (R5):** 10 minutes at 128×128, counting `framestamp` gaps. Expect 0 dropped frames, or a
  logged overrun count that explains every gap.
- **T5 (R6):** full frame and 2×2/4×4 binning: about 120 fps with no errors, and DCAM buffer memory
  within the budget.
- **T6, latency report:** `fg_timing.delta_aw` and acquisition jitter before and after, so the cost of
  draining (Q2) is measured rather than assumed.
- **T7, fastest rate without bundling:** step the ROI height down (64, 32, 16, 8 lines at 128 wide)
  and record the camera rate (`camham.fps`) and the published rate (shmimmon). This finds the height
  where the host stops keeping up without frame bundling, as a baseline for the frame-bundle plan.
  (The original T7 frame-bundle test moves to that plan.)

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

No changes until this plan is reviewed (Instructions to Agent).

1. **Diagnose first (T0).** Add temporary timing and counter instrumentation in a scratch build, not
   committed to the feature branch. Run T0 and record the results here. Experiments, one at a time,
   each measured with `acq_fps`:
   - E1: suppress the per-frame `std::cerr` skip print;
   - E2: `framegrabber.threadPrio` raised (e.g. 80) and the thread pinned with `cpuset`;
   - E3: `chrt -f -p 80 <DCAM thread>`. **On hold (Q5)**; revisit only if steps 2-3 and E1-E2 fall
     short.

   If E1-E2 alone reach R1/R2, step 2 still applies for robustness, and the results feed into the
   config defaults. The T0 counters are built into step 2 as permanent, rate-limited statistics (frames
   per wake-up, overruns, framestamp gaps), so no throwaway scratch build is needed. Each restart for a
   measurement is announced to the user first (Q4).
2. **Drain the DCAM backlog in `acquireAndCheckValid()`.** This is the main fix.
   - New documented members: `m_nextFrame` (the next frame number to publish, counted from capture
     start), `m_framesTransferred` (the last `nFrameCount`), and an overrun counter.
   - If `m_nextFrame < m_framesTransferred`, don't wait. Lock frame `m_nextFrame % m_frameCount` and
     publish it. Otherwise call `dcamwait_start( FRAMEREADY | STOPPED )`, then `dcamcap_transferinfo()`,
     update `m_framesTransferred`, and continue.
   - Overrun: if `m_framesTransferred - m_nextFrame > m_frameCount - margin`, jump to the oldest frame
     that's still valid, add the skipped count to the overrun counter, and log it through `log<>` at most
     once per second. This replaces the per-frame `std::cerr` "Skipped frame(s)!" message.
   - Check each frame's `framestamp` against `m_nextFrame` as a guard, and count mismatches.
   - Acquisition time: use the frame's DCAM `timestamp` for `m_currImageTimestamp` if T0 shows it is
     on the same clock as `CLOCK_REALTIME`, so `atime` reflects when the camera delivered each frame even
     for drained frames. Otherwise keep `clock_gettime`.
   - Reset the counters in `configureAcquisition()` after `dcamcap_start()`, so ROI changes (R4)
     restart cleanly.
   - Put the index and overrun arithmetic in a small `inline` helper so U1 can test it.
3. **Size the DCAM ring buffer by time and memory.** Replace the fixed `m_frameCount{ 10 }` with a count
   that covers at least about 100 ms at the current frame rate, capped by a byte budget.
   - Use a new config key, `camera.circBuffMaxBytes`, named like pvcamCtrl's existing key.
   - Example: 128×128 frames are 32 KiB, so 1000 frames is 32 MiB. Full-frame frames are 18.9 MB, so a
     budget of about 512 MB gives about 27 frames.
   - Keep the computed count within sensible bounds.
4. **`camham.fps` (R3).** Keep publishing `DCAM_IDPROP_INTERNALFRAMERATE`, which `configureAcquisition()`
   already refreshes on every ROI change. Verify with T1/T2. Adjust only if Q3's answer differs.
5. **Config.** Add any `camham.conf` changes found in step 1 (e.g. `framegrabber.threadPrio`, `cpuset`)
   to the deployment config, with comments.
6. **Frame bundle: moved to a separate plan (Q6).** Notes to carry over: a `camera.frameBundle` config
   key (N); set `FRAMEBUNDLE_MODE`/`_NUMBER`; read `FRAMEBUNDLE_FRAMESTEPBYTES`/`_ROWBYTES`; publish the
   N images per DCAM frame through the step 2 drain logic with interpolated timestamps; fall back to N = 1
   on `DCAMERR_FRAMEBUNDLESHOULDBEOFF`. T7's results are its baseline.
7. **Unit tests** U1/U2 (rule 20), with Doxygen groups in `tests/groups.dox` and `\ref` links per rule 21.
8. **Commits** (rules 19 and 23, short messages):
   - the drain logic and buffer sizing (functional);
   - tests;
   - config;
   - docs, including this plan's results;
   - formatting, if needed.

   Each is built with GCC 14 and clang-formatted, and pushed to `joshua-liberman/hamsci-app-cpp` when
   you ask.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed >

- **Publish bursts:** draining publishes 2-3 frames back to back after a slow wake-up. Consumers that
  wait on the semaphore see every frame, but `circBuffLength=1` readers only see the newest. Consider
  raising `circBuffLength` in `camham.conf`.
- **Stopped capture while draining:** if `DCAMWAIT_CAPEVENT_STOPPED` arrives with backlog pending,
  publish the backlog only if it's still valid, then return 1 so the framegrabber can reconfigure.
- **Counters across reconfigure and power-off:** the counters reset in `configureAcquisition()`, and
  `closeCamera()` invalidates any pending frames.
- **Binning:** frame size and rate change with 2×2/4×4. The sizing helper uses the actual
  `IMAGE_FRAMEBYTES`.
- **Row padding:** `IMAGE_ROWBYTES` and `FRAMEBUNDLE_ROWBYTES` may exceed `width × 2`. The current copy
  assumes contiguous rows. Check this with T0; it matters for bundle mode.
- **`std::cerr` skip messages** are replaced by rate-limited `log<>` messages.
- **Still open from earlier plans, out of scope here:**
  - readout-speed selection is never written to the camera;
  - `whilePowerOff()` calls `onPowerOff()`;
  - the setpoint log message reports a setpoint that isn't written.
- **USB fallback:** if the camera is ever on USB 3.1, the rates are much lower (4084 fps at 4 lines).
  Not addressed.
- **Reconnect race (crash risk):** `connect()` (main thread) tears down DCAM through `closeCamera()`
  while the framegrabber thread may still be using the handles. It caused the 2026-10-11 12:00 segfault
  in `libdcamapi.so` during an ERROR loop. It should be fixed separately, e.g. by having the
  framegrabber thread finish its DCAM calls (reconfigure or stop) before `connect()` closes the camera.
- **`libfgphnx.so` segfaults** on stop or restart in older builds (see the round 2 notes). Needs its own
  investigation.
- **Attached buffers:** DCAM 4.0.6898 refuses `dcambuf_lockframe()` on attached buffers
  (`DCAMERR_NOTREADY`) and segfaults in `dcambuf_release()` of them. Don't use `dcambuf_attach()` unless
  a later DCAM version is shown to fix this.

## Execution Notes

- 2026-10-05: the plan was reviewed and the questions answered (see "Answers"). Implementation started
  with the code-only steps (2, 3 and 7). Restarts for the hardware tests wait for the user's
  go-ahead.
- **Step 2 (drain the backlog), done, not yet hardware-tested:**
  - `acquireAndCheckValid()` now publishes every transferred frame in order. It calls
    `dcamcap_transferinfo()` first and waits on `dcamwait_start()` only when no unpublished frame is
    left.
  - Frame selection and overrun handling live in the new dependency-free `apps/orcaCtrl/orcaUtils.hpp`
    (`orcaSelectFrame()`). The ring index is computed relative to `nNewestFrameIndex`, so it doesn't
    assume capture started at ring index 0. The ring margin is 2 slots.
  - Each frame's `framestamp` is checked against its expected frame number; mismatches are counted.
  - The per-frame `std::cerr` "Skipped frame(s)!" message and `m_camera_timestamp` are removed.
    `appLogic()` logs new drops and framestamp errors as `LOG_WARNING`, at the 1 Hz FSM rate.
  - A new read-only INDI property `camham.acq_stats` publishes `frames`, `frames_per_wait`, `dropped`,
    `stamp_errors` and `buffer_frames`. This is the permanent form of T0's counters.
  - Acquisition time comes from the DCAM frame timestamp if it is within 1 s of `CLOCK_REALTIME` on the
    first frame after a reconfigure, otherwise from the host clock. The choice is logged once.
  - On power-off, DCAM failures in `updateTransferInfo()` and `dcambuf_lockframe()` aren't reported as
    ERROR.
- **Step 3 (buffer sizing), done:** `orcaBufferFrames()` sizes the ring buffer to at least 0.1 s of
  frames and at least 10 frames, at most 10,000, within the new `camera.circBuffMaxBytes` (uint32,
  default 0.5 GB, same name as pvcamCtrl's key) and never below 3. Examples: 202 frames at 128×128 and
  2013 Hz, 105 at 256×256, 12 at full frame (226 MB). `configureAcquisition()` now releases the previous
  buffer before reconfiguring. `reconfig()` only released it inside its retry loop, so it was normally
  re-allocated without being freed.
- **Step 7 (unit tests), done:** `apps/orcaCtrl/tests/orcaUtils_test.cpp` covers U1 (in-order selection,
  backlog, wrap-around, non-zero start index, exactly full, overrun by 1 and by many, invalid inputs) and
  U2 (128×128, 256×256, full frame, memory limit, min and max, never below 3). Added to `tests/tests.list`.
  `make -f Makefile.one t=../apps/orcaCtrl/tests/orcaUtils_test` result: **All tests passed (47 assertions
  in 3 test cases).** The `orcaCtrl_unit_test` group is defined in the test file with
  `\ingroup application_unit_test`, following `cred2Ctrl`.
- **Build:** clang-format is clean and the GCC 14 build succeeds, with only the existing
  narrowing/unused-`commit` warnings; the new DCAM error-logging calls follow the existing pattern. The
  first link failed because `libMagAOX/logger/logFileRaw.o` (2026-09-24) was older than
  `logFileRaw.hpp` (updated by the `dev` merge) and the libMagAOX Makefile doesn't track that header. The
  stale object was deleted and libMagAOX rebuilt. No other stale objects were found.
- **Next:** T0-T7 on hardware. This needs `sudo make install` (setuid root binary in `/opt/MagAOX/bin`)
  and an orcaCtrl restart, both announced to the user first.

### Hardware test round 1 (2026-10-11, build with steps 2-3)

Setup: the user installed the 10-05 build and restarted `camham` (11:37). The log showed `allocated 13
DCAM buffer frames of 2097152 bytes` and "DCAM frame timestamps are not on CLOCK_REALTIME; using host
time". The stream rate was measured with a read-only `cnt0` sampler (scratchpad `shmimRate`; it reads
120.3 Hz at 1024×1024 versus `camham.fps` 120.56). ROI and exposure were set through INDI. Note that
`setINDI` needs every element of a property, e.g. `camham.exptime.current=X;target=X`.

| Test | ROI | `camham.fps` | Stream (cnt0) | `acq_stats` | Result |
| --- | --- | --- | --- | --- | --- |
| T2 | 128×128 | 2012.9 | **780.3 Hz** | frames_per_wait 1526, dropped 1219, stamp_errors 555 | **fail** |
| T1 | 256×256 | 1044.3 | **1021.9 Hz** | frames_per_wait 54, dropped 0 (in 10 s) | **fail** (below 1040 ± 10; backlog growing) |
| T3 | 1024×1024 | 268.6 | 268.6 Hz | dropped 0 | pass |
| T3 | full 4096×2304 | — | **0 Hz** | — | **fail**: "ROI out of bounds" ERROR loop every ~5 s |
| T3 | back to 128×128 | 2012.9 | 776.6 Hz | dropped 1052 | recovered |

Findings, which **revise finding 3** of the original analysis:

- **The bottleneck is not frame-event latency.** With the backlog drained, the loop almost never waits
  (`frames_per_wait` 1526), yet it still manages only about 780 frames/s. Each iteration costs about
  1.3 ms with no waiting.
- The frame grabber thread uses only about 4% CPU, but makes **about 6,200 voluntary context switches
  per second at about 810 frames/s, about 7.6 sleeps per frame**. In the drain path the only DCAM call
  per frame is `dcambuf_lockframe()`. It appears to hand off to DCAM's internal worker thread, which
  runs at normal priority with about 9% CPU, and back. So **each `dcambuf_lockframe()` call costs about
  1.2 ms of blocking.** That fits with `acq_min` of about 0.6-0.8 ms and the old 760-780 Hz cap, which had
  one lockframe per frame as well.
- `stamp_errors`: while draining, the transfer snapshot (`m_framesTransferred`) is only refreshed when
  the backlog empties. During a 200-frame drain at 1.3 ms per frame the camera writes about 500 more
  frames, so slots are overwritten between selection and lock, and the framestamp check rightly flags
  them. The overrun accounting needs a fresh snapshot for every batch.
- **Full frame:** the test set only W/H, so the centre stayed at (511, 511). `setDcamRoi()` rejected the
  request as out of bounds and `configureAcquisition()` put the app in ERROR, retrying every ~5 s until a
  valid ROI was set. `checkNextROI()` is an empty stub.
- **The x centre flips on every reconfigure** (3583 ↔ 511 seen). With `defaultFlip=flipLR`,
  `setDcamRoi()` mirrors x, but `configureAcquisition()` computes `m_currentROI.x` from the raw
  `SUBARRAYHPOS` without mirroring it back. The y centre has the same issue under `flipUD`. This breaks
  R4.
- The drop and framestamp warnings logged every second while overrunning, which is too noisy.
- The camera was restored to 1024×1024 at 8.2944 ms (its state before the tests) afterwards.

### Revised implementation (round 2)

- **2a. Cache the DCAM frame pointers.** After `dcambuf_alloc()`, call `dcambuf_lockframe()` once per
  ring index and store the `buf` pointers and `rowbytes`. DCAM's ring memory is fixed until it's
  released. Publishing then reads `m_framePtrs[ringIndex]` without a DCAM call. Each batch costs one
  `dcamcap_transferinfo()`, which also refreshes the snapshot for overrun accounting, and the wait
  happens only when nothing is new.
- **2b. Verify the mapping without a per-frame cost.** About once a second, call `dcambuf_lockframe()` on
  the frame being published and check that its `buf` matches the cached pointer and its `framestamp`
  matches the expected frame number. Mismatches are counted in `stamp_errors`.
- **2c. Rate-limit the warnings** to at most one per 10 s per kind, with totals.
- **R4a. Fix the ROI centre readback** in `configureAcquisition()`: undo the flip (the inverse of
  `setDcamRoi()`) before computing `m_currentROI.x`/`.y`.
- **R4b. Implement `checkNextROI()`:** round W/H/X/Y to the manual's 4-pixel steps and clamp the size and
  centre so the ROI fits on the sensor. An out-of-range request then becomes the nearest valid ROI
  instead of an ERROR loop.
- Out of scope, still on hold: DCAM worker-thread priority (E3). Revisit only if 2a falls short.

### Round 2 implementation (2026-10-11, built, not yet installed)

- **2a:** done with `dcambuf_attach()` rather than caching DCAM-owned pointers.
  - The new `allocateBuffers()` sizes the ring with `orcaBufferFrames()`, allocates page-aligned
    app-owned memory (frame stride = frame bytes rounded up to a page, pre-faulted with `memset`) and
    attaches it as `DCAMBUF_ATTACHKIND_FRAME`. Frames are then read through `m_framePtrs[ringIndex]`.
  - If attaching fails, it logs a warning and falls back to `dcambuf_alloc()`, locking every frame as in
    round 1.
  - The attached memory is freed only when reallocating (framegrabber thread) and in the destructor,
    never in `closeCamera()` (power-off on the main thread), so the framegrabber can't read freed memory.
    `dcambuf_release()` with its default kind detaches it.
- **2b:** `dcambuf_lockframe()` runs about once a second (every `INTERNALFRAMERATE` frames). It checks
  the framestamp advance since the last verification, and that the locked `buf` equals the cached
  pointer. Both kinds of mismatch count in `stamp_errors`.
- **2c:** drops and mapping errors are combined into one `LOG_WARNING` at most every 10 s, with totals.
- **R4a:** `configureAcquisition()` undoes the LR/UD flip on the `SUBARRAYHPOS`/`VPOS` readback.
- **R4b:** the new `orcaClampROI()` (in `orcaUtils.hpp`) clamps the ROI. It is used by `checkNextROI()`
  (which now updates the INDI targets and logs a notice when it adjusts), by `setNextROI()` before every
  reconfigure, and in `configureAcquisition()` for startup and default ROIs. New class constants:
  `c_sensorWidth`, `c_sensorHeight`, `c_roiStep`.
- **Removed:** the DCAM-timestamp option. Round 1 showed DCAM timestamps aren't on `CLOCK_REALTIME` for
  this camera, so acquisition times are always host time.
- **Unit tests:** an `orcaClampROI()` test case was added (valid ROI unchanged; the round-1 full-frame
  case; size rounding and limits; corner rounding; right and bottom edge; half-pixel center).
  **All tests passed (69 assertions in 4 test cases).**
- **Build:** GCC 14 succeeds with only the existing two warning types (narrowing in DCAM error logging,
  unused `commit`), and clang-format is clean.
- **Next:** the user installs and restarts, then round 2 hardware tests T1-T4 and T7.

### Round 2 hardware attempt (2026-10-11, 12:00 MST): crash, fixed in round 2b

- The user installed the 11:49 build and restarted. The log showed **`attached 13 frame buffers of
  2097152 bytes to DCAM`**, so `dcambuf_attach()` works on this camera.
- The first verification `dcambuf_lockframe()` (frame 0) then failed with **`DCAMERR_NOTREADY` ("Status
  is not ready!")**: DCAM doesn't allow locking frames in attached buffers. That put the app in ERROR.
  `appLogic()` reconnected every ~5 s, and on the third cycle **orcaCtrl segfaulted in
  `libdcamapi.so`** (coredumpctl: PID 3770398, SIGSEGV, 12:00:30 MST). `camham` was left down.
- Likely crash mechanism: reconnecting from ERROR calls `connect()` → `closeCamera(true)` (abort, stop,
  release, close, `dcamapi_uninit`) on the main thread while the framegrabber thread is still in
  `reconfig()` or other DCAM calls. This race existed before, but the repeated ERROR loop exposed it.
  It's recorded as a follow-up; see below.
- Separately, `dmesg` shows earlier orcaCtrl segfaults in **`libfgphnx.so`** (the CoaXPress frame-grabber
  library) on 09-30, 10-01 and 10-11 between 11:24 and 11:35, all from builds before this work. They look
  like the existing crash on stop or restart, are unrelated to this change, and are recorded as a
  follow-up.
- **Round 2b fix:** with attached buffers, `dcambuf_lockframe()` is no longer called at all. Overruns
  are still detected from the `dcamcap_transferinfo()` counts. With the DCAM-owned fallback, every frame
  is still locked and its framestamp checked. `m_nextVerifyFrame` was removed, and `stamp_errors` now
  applies only to the fallback. Rebuilt with GCC 14 (only the existing warnings); unit tests: all passed
  (69 assertions, 4 test cases).

### Hardware test round 2b (2026-10-11, 12:04-12:07 MST, attached buffers)

The log showed `attached 13 frame buffers of 2097152 bytes to DCAM` and no errors. The images in the
stream are real (mean about 26,100 counts, 100% non-zero, changing).

| Test | ROI | `camham.fps` | Stream (cnt0) | `acq_stats` | Result |
| --- | --- | --- | --- | --- | --- |
| baseline | 1024×1024, 8.3 ms | 120.6 | 120.4 Hz | dropped 0 | pass |
| T2 (60 s) | 128×128 | 2012.9 | **1544.3 Hz** | frames_per_wait 1, dropped 0 | fail (was 780) |
| **T1** (20 s) | 256×256 | 1044.3 | **1044.3 Hz** | frames_per_wait 1, dropped 0 | **pass** |
| T7 (20 s) | 128×64 | 3753.8 | **1544.6 Hz** | frames_per_wait 1, dropped 0 | cap |
| T7 | 128×32 | — | 0 Hz | — | stream stopped; **segfault about 19 s later** |

- **A fixed delivery ceiling of about 1545 frames/s.** The rate is the same at 128 and 64 lines (camera
  2013 and 3754 fps). `frames_per_wait` is exactly 1 with no drops, so orcaCtrl handles every frame
  DCAM transfers. CPU use is low (framegrabber thread 2%, DCAM thread 7%). The limit is therefore about
  0.65 ms of per-frame handling *below* orcaCtrl (DCAM, driver or CoaXPress), and the camera's extra
  frames never appear in `nFrameCount`. Delivering several frames per transfer is what **frame
  bundling** is for, so **R2 (2000 Hz at 128×128) is likely not reachable without frame bundling**. That
  needs the user's decision; see Q7.
- **The crash (backtrace from the user, `sudo coredumpctl gdb 3772350`):** thread 1 (framegrabber)
  `syncEvent::setevent()` ← `qThread::destroy()` ← `dcamview::releasebuffer()` (`libdcamdig.so`) ←
  `dcambuf_release()` ← `orcaCtrl::configureAcquisition()`. The 12:00 crash was at the same instruction
  (`libdcamapi.so` +0x1de7b, fault address 0x19), from `closeCamera()` in the reconnect loop. So
  **DCAM 4.0.6898 segfaults releasing *attached* buffers**, in both cases. DCAM-allocated buffers were
  released across many reconfigures in round 1 without a crash. `False` in `reconfig()` is X11's `0`, so
  there was no double release.
- `reconfig()`'s retry loop used a `const` `acquisitionRunning` flag, so once entered it ran until the
  status read UNSTABLE, sleeping 1 s and releasing buffers each time. It may explain the 19 s gap before
  the crash.

### Round 2c implementation (2026-10-11, built, not yet installed)

- **No more `dcambuf_attach()`.** `allocateBuffers()` uses `dcambuf_alloc()` again and clears a
  per-slot pointer cache, `m_framePtrs` (all `nullptr`).
- **Per-slot pointer cache:** `acquireAndCheckValid()` calls `dcambuf_lockframe()` only the first time a
  ring slot is used, and caches its address; DCAM's slot addresses are fixed between alloc and release.
  Later frames in that slot are read with no DCAM call. About once a second (`m_nextVerifyFrame`) a frame
  is locked again: its framestamp advance is checked, and so is the cached address. Mismatches count in
  `stamp_errors`. Cost: about 1.2 ms per slot during the first pass through the ring after each
  reconfigure, so a fraction of a second of drops may be logged at high rates.
- **`reconfig()`:** the retry loop now re-reads the status on each pass and runs only while capture is
  BUSY, and it no longer releases buffers (`configureAcquisition()` does).
- Build: GCC 14 with only the existing warnings. Unit tests: all passed (69 assertions, 4 test cases).

### Hardware test round 2c (2026-10-11, from 12:24 MST, DCAM buffers + pointer cache)

The user installed and restarted; startup was clean (`allocated 13 DCAM buffer frames of 2097152 bytes`,
no errors, 1024×1024 at 120.4 Hz).

| Test | ROI | `camham.fps` | Stream (cnt0) | `acq_stats` | Result |
| --- | --- | --- | --- | --- | --- |
| **R1/T1** (60 s) | 256×256 | 1044.3 | **1044.3 Hz** | dropped 0, frames_per_wait 1.05 | **pass** |
| **R2/T2** (60 s) | 128×128 | 2012.9 | **2012.9 Hz** | dropped 6 (first ring pass), frames_per_wait 1.13 | **pass** |
| R4/T3 (10 s each) | 128×64 | 3753.8 | 2885.3 Hz | dropped 0 | runs; DCAM delivers fewer than calculated |
|  | 128×32 | 6613.8 | 2319.6 Hz | dropped 0 | **no crash** (crashed in round 2b) |
|  | 256×256 | 1044.3 | 1044.3 Hz | dropped 0 | pass |
|  | 1024×1024 | 268.6 | 268.6 Hz | dropped 0 | pass |
|  | full 4096×2304 (size only) | 120.0 | 119.0 Hz | dropped 0 | **pass** (clamped to center 2047, 1151; no ERROR) |
|  | 128×128 | 2012.9 | 2012.9 Hz | dropped 0 | pass |

- **R3:** `camham.fps` matches shmimmon to within 1 Hz at 128 lines and above (full frame 120.0 versus
  119.0).
- **T7 baseline for the frame-bundle plan:** without bundling, the stream tops out around 2900 Hz at
  64 lines and falls to about 2300 Hz at 32 lines. Frames beyond that never appear in DCAM's transfer
  count, so orcaCtrl shows no drops.
- **`stamp_errors`** grows slowly: 1 in 60 s at 256×256 (62,657 frames) and 2-3 in 60 s at 128×128
  (120,773 frames). That's about one per 65,536 frames, consistent with a **16-bit DCAM framestamp**
  wrapping, which the verification doesn't allow for. T4 (10 min) checks this.
- **T4 (shortened to about 5 min at the user's request):** four 60 s samples all at **2012.9 Hz**
  (120,773-120,774 frames each); **0 drops** over 595,515 frames since the last ROI change; OPERATING
  throughout. `stamp_errors` was 11: 2 already at 51,418 frames (before any 16-bit wrap, consistent with
  frames overwritten during the slow first ring pass), plus about 9 for the 595,515 / 65,536 ≈ 9.1 wraps.
  This **confirms the 16-bit wrap** explanation.
- The camera was restored to 1024×1024 at 8.2944 ms, center (511, 511), at 120.7 Hz. The center stayed
  at 511, so the flip fix holds.

### Round 2d: framestamp wrap fix (2026-10-11, built, not yet installed)

- New `orcaFramestampAdvanceOk()` in `orcaUtils.hpp` compares the framestamp advance with the
  frame-number advance modulo 2^16. `acquireAndCheckValid()` uses it for verification. A new unit test
  case covers a normal advance, a 16-bit wrap, a 32-bit wrap, and wrong advances.
- Unit tests: **all passed (74 assertions in 5 test cases)**. GCC 14 build with only the existing
  warnings.
- Expected effect: `stamp_errors` stays at 0 in steady state. A small count right after an ROI change
  at high rates (first ring pass) can still appear, and is real (overwritten frames).
- **Verified on hardware (2026-10-11, 12:42 MST, user installed round 2d):** at 128×128, `stamp_errors`
  went from 2 (at 17,658 frames, just after the ROI change) to 2 (at 219,181 frames). That's **0 new
  errors over 201,523 frames (about 3 wraps)**, with 0 drops, 2012.9 Hz, and OPERATING throughout.
- **Remaining first-pass glitch:** the 2 errors right after the ROI change came with **0 counted
  drops**. During the first ring pass each slot costs about 1.2 ms to lock, the backlog grows, and the
  transfer snapshot goes stale, so a slot can be overwritten before it's locked without being detected
  as an overrun. The frame published from it is then out of order. This is a brief silent glitch for
  R5, only right after an ROI change at high rates. Follow-up: refresh `dcamcap_transferinfo()` before
  locking a not-yet-cached slot, or count a framestamp mismatch as a drop.

### Status against requirements (round 2c hardware, 2026-10-11)

- **R1** (256×256, 1040 ± 10 Hz): **pass**, 1044.3 Hz.
- **R2** (128×128, 2000 ± 10 Hz): **pass**, 2012.9 Hz.
- **R3** (`camham.fps` within 10 Hz of shmimmon): **pass** at 128 lines and above (worst case full frame,
  120.0 versus 119.0). Not met below about 128 lines without frame bundling (see the T7 baseline).
- **R4** (seamless ROI switching): **pass** across 128×64 → 128×32 → 256 → 1024 → full (size only) → 128.
- **R5** (no silent drops): **pass in steady state** (0 drops, 0 framestamp errors over 200k+ frames).
  Caveat: right after an ROI change at high rates, the first ring pass can publish about 2 frames out of
  order without counting them (see the round 2d notes). Follow-up recorded.
- **R6** (no regression at full frame): **pass**, 119.0 Hz at full frame with a 12-frame buffer. Binning
  wasn't tested.

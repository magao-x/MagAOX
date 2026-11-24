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
ocam2KCtrl is subject to getting a frame or more behind.  This manifests as loop stability problems, and is fixed by reconfigure.  We should be able to detect this.

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->

Frame number isn't detecting it.  

The most likely detectable signature is a time of acquisition or publishing > 1/FPS later than expected.  It might also be a gradual buildup, i.e. the sum of a delays gets > 1/FPS late.

We need to look at atime, which is what the SDK gives us, and actual time at which we get the frame, and also write time.  These are three different times that might catch what is going on.

We can in principle dig out some telemetry but it depends on having data being saved at full rate while a reconfigure become necessary.  Better would be to implement a rolling diagnostic history that dumps if reconfigure is requested.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->

- track delta atime
- track delta actual arrival time
- track delta write time
- warn if any single one of those is greater than expected
- track integrated time for each over N samples and telemeter it.  Start with FPS samples
- warn if any of those sums is greater than expected
- warning in logs
- also with switches that latch if warning is in place and clear when reconfigured
- tie into labrules and rtimv_camwfs.conf
- for initial testing have a disable toggle for each so we can turn it off if the warnigns are spurious.
- have a 120 second rolling diagnostic history of times and framenumbers that is dumped on reconfigure.  Even this might not be long enough though.

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->

## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

## Task Summary

Detect camwfs falling a frame behind even when OCAM frame numbers remain consecutive. Compare the cadence at EDT DMA completion, SDK return, and main stream write. Report one-frame delays and gradual drift, and save the history preceding an explicit operator reconfigure request. Propose 15 minutes of history because recognizing the problem and requesting reconfiguration can take longer than two minutes.

The three endpoints already have well-defined capture sites:

| Endpoint | Source | Use |
| --- | --- | --- |
| `atime` | The EDT SDK supplies the DMA-completion timestamp in `pdvAcquire()`. | Marks when the frame reached host memory. |
| Arrival | `waitReturn` is immediately after `pdv_wait_last_image_timed()` returns, before the timeout query and descrambling. It uses `CLOCK_MONOTONIC`. | Shows when the application got the frame. |
| `writetime` | `fgThreadExec()` stamps the main stream with `CLOCK_REALTIME` after the image load and staged copy, before posting the stream semaphore. | Shows when the frame was ready to publish. |

The timing event carries these endpoints, the camera frame number, and the stream counter. The workstation camwfs configuration has a two-second `fg_timing` window, 20,000 timing slots, and `stageTiming=false`. Enabling the delay monitor captures the SDK-return timestamp with that configuration. The SDK timestamp contract and the positions of the stage markers are documented in the [frame-grabber timing notes](../../../libMagAOX/app/dev/doc/frameGrabberTiming.md). See [edtCamera.hpp](../../../libMagAOX/app/dev/edtCamera.hpp), [frameGrabber.hpp](../../../libMagAOX/app/dev/frameGrabber.hpp), and [ocam2KCtrl.hpp](../../../apps/ocam2KCtrl/ocam2KCtrl.hpp).

## Key Assumptions

- Use the finite positive effective FPS from the existing camera readback or external synchronization subscription. A zero internal target selects automatic FPS. Wait for the reported rate in that case. Carry the rate with its acquisition epoch so a later INDI update cannot change how queued samples are interpreted.
- Set the nominal period `T = 1 / FPS`. For each endpoint, the single-frame excess is `t[i] - t[i-1] - T`. Use the most recent `N = max(1, round(FPS))` consecutive intervals for the integrated check. The measured sum is `t[i] - t[i-N]`, and the signed excess is that sum minus `N*T`. Thus `N` intervals require `N+1` timestamps. The signed sum lets an early frame cancel part of a previous late interval.
- The inclusive excess threshold is `frameDelay.thresholdFrames*T`, with a default fraction of `1`. This triggers at a single interval of `2*T` or a rolling sum of `(N+1)*T` at the default. Round the total threshold to the nearest integer nanosecond and accept an elapsed span at or above it. Use each endpoint's own clock for the interval calculations.
- Reuse the project's header-only application pattern and timing handoff. Put the detector and history logic in an OCAM helper that can be exercised without the SDK. Keep the shared timing changes limited to the optional capture and delivery needed by that helper.

### History cost

A proposed 48-byte metadata row holds three 64-bit timestamps, a 64-bit attempt sequence, a 64-bit stream counter, a 32-bit camera frame number, and 32 bits for outcome and validity. Store the mode, FPS, stream identity, and loss counters in separate records keyed by sequence. At a constant frame rate, the following estimates apply to both steady history RAM and one uncompressed binary dump, before allocation overhead and capacity margin.

| FPS | Rows in 15 minutes | At 48 bytes per row | At 64 bytes per row |
| ---: | ---: | ---: | ---: |
| 2,000 | 1,800,000 | 82 MiB | 110 MiB |
| 3,620 (configured bin2 maximum) | 3,258,000 | 149 MiB | 199 MiB |
| 5,962 (highest configured mode) | 5,365,800 | 246 MiB | 328 MiB |

At 48 bytes, history receives about 94 KiB/s at 2,000 FPS and 279 KiB/s at 5,962 FPS. This is metadata storage bandwidth. Assess timing impact separately for the SDK-return clock read, the event handoff, and disk I/O. Five full-rate 48-byte dumps take about 1.2 GiB of disk. A snapshot can share sealed history chunks with the writer and release them as it writes. Budget up to about 492 MiB plus margin at 5,962 FPS if a stalled writer pins one full 48-byte snapshot while a new history accumulates.

## Requirements

- Evaluate all three single intervals and all three rolling sums from continuous frames. Telemetry should include the configured FPS and sample count, measured spans and excesses, validity, warning and enable bits, and capture loss. Record the measurements even while a warning is disabled.
- Provide six independently disableable warning checks in INDI, one per endpoint and test. Emit one `LOG_WARNING` the first time each enabled check latches. Include the endpoint, observed interval or sum, expected value, FPS, and frame number. Publish read-only latch switches and clear them after acquisition successfully restarts following reconfiguration.
- Feed the enabled, latched warnings into labrules at warning priority and display them through the existing rtimv warnings overlay. Disabling a check suppresses its operator warning immediately. Re-enabling it restores any warning that had already latched.
- Retain a configurable history of endpoint timestamps and camera frame numbers, including available data from rejected and failed attempts. Use 900 seconds as the proposed default. An explicit operator request through `camwfs.reconfigure.request` is the sole dump trigger. Save the history up to that request on a non-real-time thread, including the run identity, rate and mode, raw clocks, stream counter, attempt sequence, outcome, and capture losses needed to interpret it.

## Questions and Points of Clarification

1. Should exactly one period of excess trigger a warning? The proposed inclusive threshold catches an interval of exactly `2*T`. A strict threshold would require more than a full frame of excess.

ANSWER: Let's use >= rather than >.  But also let's make this a configurable fraction of a full frame, e.g. 0.5*T or 0.99*T, with default 1.

2. Is a configurable `frame-delays` subdirectory under the camera's log directory suitable for the dumps? The proposed default keeps the five most recent successful dumps per camera. Confirm any collection path or retention requirement that should override this.

ANSWER: yes.

## Tests

- Feed deterministic timestamps for nominal cadence, ordinary jitter, a single frame-period delay, and gradual drift that crosses only the rolling threshold. Exercise each endpoint independently, the exact threshold and adjacent representable times, signed cancellation, fractional and changing FPS, automatic/unknown rates, and rolling-window warmup and eviction.
- Feed an unsigned frame-number wrap, rejected frame, missing or reversed timestamp, camera/mode restart, and a diagnostic handoff gap. Verify that only the affected baselines reset and missing diagnostic samples cannot masquerade as late frames.
- Extend the mocked OCAM and framegrabber tests to cover arrival capture with stage timing off, placement relative to timeout/rearm, and the existing write stamp and main/sync post order. Exercise disable, re-enable, latch and log behavior, failed reconfiguration, successful restart, and queued pre-reconfiguration frames. Verify that another app with the monitor disabled retains its existing timing behavior.
- Test history wrap, the 900-second cutoff, a manual request during an in-flight frame or automatic reconfiguration, and repeated presses while a dump is pending. Verify that mode, FPS, synchro, and automatic recovery paths generate zero dumps and retries generate one dump for the original manual request. Test ingress and writer saturation and a failed file write, then verify chronological output and continued acquisition. Round-trip the compact file and new telemetry, including the telemetry text and JSON output. Parse the companion rules with the existing stateRuleEngine configuration tests and exercise switching the rtimv warning on and off using synthetic properties.
- Have the operator A/B capture off and on at the same camera mode, FPS, loop load, and staged-copy settings. Compare `fg_timing.delta_aw`, write-interval jitter and tail, timing handoff losses, and recovery time during a dump. Include a sustained run at the highest deployed rate. Confirm that a lag event latches a warning, a manual reconfigure saves the preceding history, and the warning clears after the camera resumes.

## Implementation Plan

1. Add an opt-in camwfs monitor. Reuse `frameGrabberTiming`'s event, sequence, and fixed SPSC handoff. When only the monitor is enabled, capture the single `waitReturn` marker at the existing EDT hook and reuse the normal `writetime`. When stage timing is already enabled, share its marker. Deliver raw events to the OCAM consumer before the existing timing window filters by session or age. Keep semaphore order, staged copy, DMA rearm, and the legacy `fg_timing` and stage telemetry paths as they are. Validate that paired capture is available when the monitor is enabled.
2. Keep the delay calculation and history on the consumer side. Use the attempt sequence, stream identity and counter, camera number, and acquisition epoch to establish continuity. Compare the first interval as soon as two valid observations exist, and use timestamps `N` intervals apart for the full rolling sum in constant time. Leave missing timestamps invalid and reset only the affected endpoint if one timestamp is missing. Reset all three baselines on a frame/attempt gap, acquisition restart, or effective FPS change. Preserve the raw observations in history across these boundaries. Record the first crossing and the integrated values through the existing OCAM `dev::telemeter` path at its usual cadence, with an immediate record at a latch transition. Add an app-specific FlatBuffer type so existing camera and framegrabber log readers can keep reading their existing records.
3. Add `[frameDelay]` configuration and register the warning enables and latches at startup. Use an optional app hook at `stdCamera::newCallBack_reconfigure()` to tag the explicit INDI request before the worker sees the reconfiguration flag. Give the manual request its own ID and monotonic cutoff so it remains identifiable if the worker coalesces requests or retries a failed reconfiguration. Use an ordered timing boundary with the producer's last applicable sequence to determine when the consumer has drained the tail. Keep the control notification independent of per-frame queue saturation and report any lost samples. A successful acquisition restart clears the warning latches. Keep the latches through `pdvReconfig()` and acquisition startup failures, and consume old timing events in order so they cannot relatch the new run.
4. Store compact metadata in fixed-size chunks allocated and recycled by the ordinary consumer. Bound total history capacity by duration times the highest configured mode FPS plus margin for attempt bursts, and expire rows by monotonic time. On a manual request, hand the writer the sealed chunks covering the requested interval, with offsets for the first and last rows. Share those chunks with the live history and release them as the writer progresses. Allow one in-flight snapshot and report when a second request cannot be written. Write a versioned binary file with integer nanoseconds and document offline CSV export. Write to a temporary file in the destination directory and rename after a successful close. Report shortened coverage, writer backpressure, and filesystem errors through the normal logs. Keep filesystem work outside the acquisition thread and camera/INDI locks.
5. For deployment, enable `[frameDelay]` in `camwfs.conf`. To show the same warnings in rtimv, add `swVal` checks for the enable and latch states, combine them with `ruleComp` warning rules, and add the rule names to the existing `[rules] warnings` list in `rtimv_camwfs.conf`. The warnings plugin reads `labrules.warning.<rule>` from INDI.

### Implementation and verification

- Implemented the independent warning switches, inclusive `thresholdFrames` setting, nanosecond cadence telemetry, and manual-only history. The [operator notes](../../../apps/ocam2KCtrl/doc/frame-delays.md) document the dump format and CSV exporter. The optional companion configuration was tested in a temporary copy of the configuration checkout.
- Built `ocam2KCtrl` with the archived EDT SDK on `mx11`. Five offline suites passed, covering the cadence helper and logger, OCAM reconfiguration and controller paths, framegrabber timing, and framegrabber publication integration. Total coverage was 1,911 assertions across 74 test cases.
- Loaded 142 rules using the production labrules parser, checked all six camwfs rules with synthetic INDI switch states, and confirmed the six warning keys in the rtimv configuration. The CSV exporter preserved clock and epoch data and rejected a truncated file.
- A synthetic 900-second history at 5,962 FPS wrote all 5,365,800 rows in 257,558,536 bytes (246 MiB). The isolated test peaked at about 255 MiB of RSS. This measurement covers the history helper and writer on the workstation. Instrument A/B timing remains open.
- Added `frameDelay.warnings=false` to start with all six warning enables Off while continuing cadence telemetry and manual dumps. The default keeps all six enabled at startup, and the existing INDI switches can still change individual enables during operation.

## Follow-up and Edge Cases

- A stream already a fixed number of frames behind can keep a normal cadence on all three endpoints. When all three clocks are kept in the history, changes in same-frame timestamp separation can still help locate a growing backlog. A fixed preexisting offset would need a reference from an unaffected run or the external trigger for detection.
- A small error in the effective FPS can trip the rolling check by itself. At a nominal 2,000 FPS, one frame of excess over a one-second window corresponds to about 0.05% rate error. Check steady runs in both internal and external synchronization modes before relying on the rolling alarms.
- A forward wall-clock adjustment can look like a late interval in `writetime` and possibly SDK `atime`. Keep the monotonic arrival series available for comparison and flag suspected clock discontinuities in the dump. Confirm SDK clock provenance on the deployed EDT driver before using absolute differences as latency measurements.
- Lossy handoff, process termination, or a stalled writer can shorten or prevent a dump. Label the available coverage and losses. Qualify polling and disk I/O under loop load, and measure memory at the deployed frame rates and during a manual dump.

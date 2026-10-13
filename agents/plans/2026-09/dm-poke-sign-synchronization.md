# DM poke sign and synchronization investigation

Status: investigation and hardware validation pending, 2026-09-26. This plan records the June 17 discussion and a fresh source review. It does not establish the cause of the intermittent sign flip.

## Reported behavior and prior decisions

- On the instrument, `dmPokeXCorr` sometimes publishes an `m_pokeImage` with the opposite sign from the expected poke response. A run can appear normal or inverted; changing the sign of `poke_amp` has been used to compensate after detecting an inverted run. This is an observation from the June 17 conversation, not a reproduction on this workstation.
- The reported configuration in `/opt/MagAOX/config/twAlign-camwfs-wfs.conf` uses `camwfs`, `dm01disp06`, `pokeAmp=0.2`, `dmSleep=1000` microseconds, `nPokeImages=5`, and `nPokeAverage=500`. It does not set either new option, so this branch defaults to trigger stream `dm01disp` and `nSettleImages=1`. Confirm the deployed configuration and live INDI values before testing.
- `dm01disp06` is intentionally written in passive mode. The separate `dmcomb` process sums it with the other DM channels and publishes `dm01disp` when a channel counter advances. The physical DM responds downstream of that combined stream. The source reviewed here is `../milk-xwcl/plugins/cacao-src/AOloopControl_DM/AOloopControl_DM_comb.c` at local commit `336160d0`; confirm the version running on the instrument.
- The June 17 work created branch `jrmales/dm-poke-sync`: `42a064fd` locks the WFS image during accumulation; `38674675` waits on the combined DM stream; `496a8bc2` adds configurable settle frames. The branch is clean and pushed. As of this review it has three commits ahead of its June base, is 42 commits behind `origin/dev`, and has no PR. The new path has not been validated on the instrument.

## Current event path

1. `basicTimedPoke()` sets the four actuator values to `pokeSign * m_poke_amp`, flushes its `dm01disp` semaphore, and writes the passive `dm01disp06` image (`libMagAOX/app/dev/dmPokeWFS.hpp`, around lines 987-1004).
2. It accepts the next `dm01disp` semaphore wake, sleeps for `dmSleep`, flushes the local WFS semaphore, discards `nSettleImages` WFS wakes, and accumulates `nPokeImages` WFS images with a sign determined by `pokeSign` (around lines 1006-1050). The final image divides the signed sum by `2 * nPokeImages * nPokeAverage` (around line 1117).
3. The WFS monitor copies a frame into one shared `m_rawImage` buffer and posts `m_imageSemaphore`. The poke thread takes `m_wfsImageMutex` before reading that buffer (around lines 806-840 and 1037-1047).
4. The installed `mx::improc::milkImage` passive `post()` clears the write flag and posts semaphores without advancing `cnt0`. In the reviewed `dmcomb` source, a change in the sum of channel `cnt0` values starts an update; `update_dmdisp()` reads channel arrays directly, then the combined stream is published. The summation does not check each channel's `write` flag or associate the output with a particular passive write.

## Findings and limits

| Finding | What it means | Confidence |
| --- | --- | --- |
| The `dm01disp` semaphore wake is an output event, not an acknowledgment of this poke. | A `dmcomb` update already in progress can publish after the flush while having sampled `dm01disp06` before or during the new write. Other active channels can also cause the first wake. The output's actuator values must be inspected to prove inclusion of the commanded sign. | Source-supported race; frequency on hardware unknown. |
| The mutex prevents simultaneous access to `m_rawImage`, but the local semaphore does not preserve a frame snapshot. | If several WFS posts queue before the worker reads, repeated wakes can all read the same latest buffer. The current code records neither WFS `cnt0` nor a per-frame timestamp with each accumulated image. | Source-supported; actual backlog unknown. |
| A single wrong boundary frame in each five-frame poke group would mainly reduce the response, assuming comparable frame amplitudes. | A near-complete sign inversion across 500 averaged pairs suggests a sustained phase/label error, an unexpected command sign, or another systematic effect. This is an inference, not a diagnosis. | Reasoning from the reported configuration. |
| A combined command update precedes the physical DM response and the camera observation. | `dmSleep=1000` microseconds plus one discarded WFS frame may or may not cover the downstream delay. The needed number of frames must be measured on the instrument. | Pipeline timing unknown. |
| The trigger stream is closed before the WFS worker thread is joined in `appShutdown()`, and `allocate()` can close and reopen it. | A worker waiting on the stored semaphore pointer may overlap stream teardown or replacement. Review stream lifetime before sustained testing or a PR. This is a separate reliability risk, not an established sign-flip cause. | Source-supported ordering; overlap not observed. |
| Existing `dmPokeXCorr` test coverage is a construction-only placeholder. | The branch builds, but no test exercises poke-to-output association, frame identity, or sign accumulation. | Confirmed in `apps/dmPokeXCorr/tests/dmPokeXCorr_test.cpp`. |

The source also permits two `dm01disp` publications for one update when astrogrid mode with a nonzero delay is enabled. Check the live `dmcomb` settings before treating one semaphore wake as one final command. The reviewed source and local configuration may differ from the deployed system.

## Hardware test sequence

1. **Record the baseline before changing timing.** Note the running `dmPokeXCorr`, `dmcomb`, and DM-controller versions; effective config and INDI values; WFS rate; trigger stream name; astrogrid settings; and the expected sign of the four poke actuators. Confirm that `dm01disp06`, `dm01disp`, and `camwfs` are the actual connected streams. Capture a known-good and, if possible, a failing `m_pokeImage` from the same configuration. Do not use a compensating negative `poke_amp` during the diagnostic run.
2. **Classify where sign diverges.** For each positive and negative poke transition, capture a sequence identifier and local timestamp at passive write, `dm01disp06` actuator values, `dm01disp` `cnt0` and actuator values at the accepted wake, and the WFS `cnt0`/timestamp and signed response for every discarded and accumulated frame. Existing stream recorders may provide part of this; if not, add a bounded diagnostic trace rather than printing from the image hot path. Preserve the raw frames needed to recompute `m_pokeImage` offline.
3. **Use the trace to separate mechanisms.** If the first accepted `dm01disp` frame lacks the requested poke, investigate the acknowledgment race or `dmcomb` scheduling. If combined commands have the expected sign but the corresponding WFS response lags or has the opposite sign, investigate physical/camera latency and optical sign convention. If combined command and WFS response agree but `m_pokeImage` is inverted, inspect frame reuse, accumulated-frame labels, and changes to live `poke_amp` during the run.
4. **Vary one timing parameter at a time.** Repeat single measurements with `nSettleImages` of 1, 2, and 3, retaining `nPokeImages=5` and a positive amplitude. If needed, test a longer `dmSleep` separately. Record the signed response per poke group, missed/duplicate WFS counters, and inversion frequency, rather than judging only the final cross-correlation. Include a stop/restart and stream-recreation trial after the basic path is characterized.
5. **Turn a confirmed mechanism into a focused change.** Candidate changes include verifying the combined output's poke pixels and counter advance before accepting it; carrying WFS frame data plus counter/timestamp as one queued sample; and joining or stopping the worker before closing the trigger stream. Choose only after the trace identifies the failing boundary. Add a hardware-free test for the selected invariant, then rerun the app build and the relevant live sequence.

## 2026-10-13 observations and operator safeguard

Instrument observations reported by the operator:

- The state is bistable and stable: a run is either inverted or not. When inverted, `m_pokeImage` is the exact negative of `refIm`, not a distorted image.
- Restarting `twAlign-camwfs-wfs`, opening and closing the loop with it running, and resetting, initializing, and power cycling the DM did not change the state.
- In the non-inverted state, `poke_amp` up to 0.8 changed the poke image but never flipped its sign, so pyramid phase wrapping is ruled out.
- Inversions also occur when the loop controls too few modes to correct single-actuator pokes, so closed-loop overshoot alone cannot explain them.

Implications: random races and frame reuse should degrade the response rather than produce a stable exact negative, so the synchronization changes above are unlikely to be the fix. The flip is held in state that survives the tested actions; candidates include the `aol1_zrespM` reference (unchanged by any tested action) and processes that were not restarted (`dmcomb`, the DM driver, `camwfs`). Next checks: save `_poke` and `_refIm` in both states to see which one changed sign; while inverted, set `dmSleep=100000` and `nSettleImages=50` to exclude timing; correlate flips with calibration loads, camera mode changes, and process restarts.

Safeguard added on this branch:

- `pokecen.maxDelta` (default 10 pixels; `<= 0` disables) rejects a measurement when either delta component exceeds it. An inverted response correlates to a distant sidelobe because the cross-correlation peak search is global, so it is caught.
- A rejected measurement is not published. `dmPokeXCorr` logs `LOG_ERROR` suggesting a change of `poke_amp` sign, reports the normalized zero-lag correlation with the reference (near -1 when inverted), and stops measuring so the alignment loop receives no bad delta.
- Diagnostics go to `pokecen.diagnosticsDir` (default `<basePath>/rawimages/<configName>/YYYY_MM_DD/`): FITS images `_poke`, `_raw`, `_dark`, `_dmchan`, `_dmcomb`, and `_refIm`. Each header records the deltas, cross-correlation peak and zero-lag value, poke configuration, WFS fps, stream names, `cnt0` of the WFS, poke-channel, and trigger streams, combined-DM updates and WFS frames received during the measurement, and the measurement start and end times. Those times can be matched against `camwfs` stream-writer archives.
- `dmPokeXCorr_test` covers the range check. `writeDiagnostics()` needs live shared-memory streams and is not unit tested.

## Current verification and exit criteria

- On this workstation, `make -C apps/dmPokeXCorr -j2` succeeded on this branch and left the tracked working tree clean. `git diff --check` for the three branch commits passed. These checks do not validate hardware timing.
- The issue is resolved when repeated runs with a fixed positive amplitude produce the expected signed `m_pokeImage`, and captured transitions show which combined command and WFS frames contributed to each poke group. Document the tested rates, settle count, DM delay, and any remaining intermittent failures before opening a PR.
- Rebase or merge current `dev` only after preserving a reproducible baseline; the changed header has not been touched on `dev` since the branch point, but the rest of the integration still needs validation.

# accelSim zero-acceleration ADC output

2026-11-12, `parkertjohnson1/accelSim`, base `07e09306`.
The user explicitly requested implementation and local tests, with no commits,
application launches or instrument access. Host checked: `ASTR-LT-428.local`.

## Implementation

- New startup-only boolean `sim.zeroOutput`, defaultfalse. The exact INI setting
  is `[sim]` followed by `zeroOutput=true`; no INDI callback or live toggle is
  added. The distributed example retains `zeroOutput=false`.
- `accelSim::loadImageIntoStream()` selects the outgoing pair through a small
  transport-independent helper in `accelSimOutput.hpp`. Normal output returns
  the acquired ADC counts unchanged. Zero output reuses the existing
  `adcCounts(0,...)` encoder with local clamp counters, returning the same
  calibrated code for both axes:2048,2048 with the current calibration. Custom
  zero points retain existing rounding/clipping semantics.
- The helper takes the physical sample/calibration by const reference. It draws
  no random numbers, advances no model state, updates no model clamp counters
  and accesses no streams. The acquired sample remains available unchanged to
  `frameGrabberPostPublish()` and its existing DM disturbance synthesis.
- The mode changes only the four payload bytes copied to the frameGrabber
  destination. Geometry/type, pacing, frame counters, timestamps, semaphore
  publication, synchronization, DM controls and DM publication remain unchanged.
  Zero ADC telemetry does not disable or zero an enabled DM disturbance.
- Application/test Makefile dependencies include the new header. Source edits
  follow existing documentation and formatting conventions; newly edited control
  bodies use explicit braces. No physical-model, DM or framework source is changed.

## Validation

The source worktree is outside the session's writable roots. A source-identical
temporary copy under `/private/tmp/accelsim-zero-jnfjqpa7` was used to prepare and
test the patch before requesting the filesystem approval needed to apply it.
No native application, service, instrument stream or DM was accessed.

| Check | Result |
| --- | --- |
| Combined model, synchronization, physical projection and mock DM runtime C++ suite, C++20/O3/strict FP |36 cases,308607 assertions passed |
| Same portable suite with ASan/UBSan, O1/strict FP |36 cases,308607 assertions passed; no sanitizer findings |
| Existing Python PSD regression suite |4 tests passed using `/opt/anaconda3/bin/python` |

The system `/usr/bin/python3` lacks NumPy; the existing Anaconda interpreter has
the required numerical packages. No dependency installation was performed.

New portable cases check default/explicitfalse output,2048,2048 output, custom
offset rounding and clipping, and8,000 paired generated frames. Every selected
zero-output pair is constant, while physical/WFS pointing, noisy acceleration,
underlying ADC values, clamp counters and future RNG progression match the normal
model exactly. Motion and normal ADC variation remain nonzero.

A hidden native `[accelSim-zero-streams]` case checks actual frameGrabber
publication at temporal depths1 and4. It loads the real boolean config, verifies
its private ImageStreamIO namespace before any image access, checks every observed
published pair, and requires advancing counters/acquisition timestamps and matching
per-plane acquisition/write metadata. Initial counter0 allocation is excluded.
The existing full-app config test also asserts the new option defaultsfalse.

**The native case and full application have not been compiled/run on this Mac:**
the native ImageStreamIO/MagAO-X dependency environment is unavailable. On a
provisioned isolated Linux development host, build the full test as described in
the README and run the hidden case in a fresh process. Portable numerical passes
are not evidence of native publication or a real-time rate guarantee.

## Files

- `apps/accelSim/accelSim.hpp`
- `apps/accelSim/accelSimOutput.hpp` (new)
- `apps/accelSim/Makefile`
- `apps/accelSim/config/example.conf`
- `apps/accelSim/tests/accelSim_test.cpp`
- `apps/accelSim/README.md`
- `tests/Makefile.one`
- This engineering record.

The user will review, stage and commit the changes. No commit/push is authorized.

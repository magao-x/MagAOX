We need to add a 2nd ImageStreamIO stream as an output of ocam2KCtrl.  It will be an empty, or minimum size, stream.  The reason to do so is to have a 2nd meta-data, namely semaphore, only stream structure that can be used to synchronize operations across computers.  It should be initialized, prepared, and maintained as part of the main dev::framegrabber setup, and the semaphores should be incremented immediately after the main stream semaphores are incremented.

As a 2nd pass to this effort, we need to add tests for ocam2KCtrl.  Currently only the ocamUtils parsers are tested.  Please follow the standards recently developed for flowRPM and adcTracker and elliptecCtrl (the last two are on feature branches).

Review AGENTS.md, then lease update this document with a plan, below.  Do not alter this prompt, and do not being executing until I have reviewed the plan.  Use the feature branch jrmales/ocam2k-sync.

Plan:
1. Review the `ocam2KCtrl` framegrabber lifecycle and the `dev::frameGrabber` ownership model to identify the smallest app-local change that adds a second `ImageStreamIO` stream without changing existing image-stream behavior.
   - Add app-local state for the sync-only stream name, `IMAGE` handle, ownership, and helper routines to create, resize, initialize, and destroy the secondary stream.
   - Keep the secondary stream prepared and maintained in the same startup, reconfigure, power-off, and shutdown phases as the main framegrabber stream.

2. Wire sync-stream semaphore publication into the acquisition path.
   - After the main stream metadata update and semaphore post complete, update the sync-only stream metadata needed for consumers and post its semaphores immediately afterward.
   - Use a `1x1` `uint8` sync-only stream so the stream stays minimal while remaining well-formed for downstream applications, and keep that choice explicitly documented in code.

3. Add `ocam2KCtrl` unit tests following the newer application-test patterns used by `flowRPM`, `adcTracker`, and `elliptecCtrl`.
   - Create `apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp`.
   - Use the `libXWCTest` namespace structure, a dedicated Doxygen test group, `tests/testXWC.hpp`, and a focused test harness class that exposes only the protected helpers needed for test coverage.
   - Keep `ocamUtils_test.cpp` for parser coverage and add new tests for app-level configuration, helper logic, and any sync-stream behavior that can be exercised without camera hardware.

4. Update test registration and Doxygen grouping.
   - Add the new `ocam2KCtrl` test target to `tests/tests.list`.
   - Update `tests/groups.dox` as needed to provide the `application_unit_test` Doxygen group expected for application tests.
   - Bring touched test files up to the current Doxygen style across the full changed file, not just the new lines.

5. Verify and document the implementation.
   - Run `clang-format` on all touched files.
   - Build and run the targeted `ocam2KCtrl` tests, plus any nearby regression checks needed for confidence.
   - Update this plan file with implementation notes if the final design differs from the outline above.

Resolved convention note:
- For MagAOX applications, use `application_unit_test`; `app_unit_test` is reserved for the `libMagAOX` app namespace rather than end applications.
- For this change, the secondary sync stream should be implemented as a `1x1` `uint8` ImageStreamIO stream rather than a zero-sized or otherwise special-case buffer.

Implementation notes:
- Added a small optional `dev::frameGrabber` post-publication hook so `ocam2KCtrl` can post the sync stream immediately after the main framegrabber stream semaphore post.
- Implemented the sync stream in `ocam2KCtrl` as a dedicated `1x1x1` circular-buffer ImageStreamIO stream with `uint8` pixels, `cnt1 = 0`, and mirrored `atime`, `writetime`, and `cnt0` metadata from the main stream.
- Added `framegrabber.syncShmimName` config support in `ocam2KCtrl`; when omitted it defaults to `framegrabber.shmimName + "_sync"`.
- Kept the sync-stream lifecycle local to `ocam2KCtrl` with helper methods to create, validate, repair, publish, and destroy the stream.
- Dropped the initial per-frame sync-stream mutex after reviewing the call graph: `configureAcquisition()` and `frameGrabberPostPublish()` both run on the framegrabber thread, and `destroySyncStream()` runs only after `frameGrabber::appShutdown()` joins that thread, so a null check is sufficient on the hot path.
- Added `apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp` covering sync-stream creation, sync-stream publication metadata and semaphore behavior, and selected hardware-free app helpers.
- Added a local test-only `apps/ocam2KCtrl/tests/edtinc.h` stub header plus OCAM/EDT stub definitions in the test translation unit so the new unit test can build without the real EDT SDK.
- Registered the new test in `tests/tests.list` and updated `tests/Makefile.one` so `ocam2KCtrl_test` can use the local EDT stub while still compiling the real `dev::edtCamera` path.

Verification notes:
- `clang-format -i libMagAOX/app/dev/frameGrabber.hpp apps/ocam2KCtrl/ocam2KCtrl.hpp apps/ocam2KCtrl/tests/edtinc.h apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp`
- `make -B -f Makefile.one t=../apps/ocam2KCtrl/tests/ocam2KCtrl_test.cpp COVERAGE=1` from `tests/`
- `../apps/ocam2KCtrl/tests/ocam2KCtrl_test` from `tests/` with shared-memory write permission: passed
- `make -B -f Makefile.one t=../apps/ocam2KCtrl/tests/ocamUtils_test.cpp COVERAGE=1` from `tests/`
- `../apps/ocam2KCtrl/tests/ocamUtils_test` from `tests/`: passed

# xInstGraph F02: checked, atomic graph updates

## Objective

Address F02 in the [xInstGraph review](xInstGraph-review.md). Each successful INDI callback should publish one complete, parseable draw.io snapshot. A failed write must be reported and must not truncate or replace the last published snapshot. Preserve the F01/F03 rules: a destination that refers to the input is never written, and `graph.clobberOutput` controls replacement of a pre-existing file only at startup.

## Evidence and boundary

- `instGraphXML::stateChange()`, `valuePut()`, and `valueExtra()` mutate the XML and call `m_doc->save_file(m_outputPath.c_str())` without checking its result (`/home/jrmales/Source/instGraph/src/instGraphXML.cpp:833-996`). `instIOPut::state()` can call `stateChange()` before it finishes propagating to outputs and beams (`src/instIOPut.cpp:163-197`). One node handler can also call several `valuePut()` and `valueExtra()` methods. A reader can therefore see an intermediate state or a truncated file.
- F01/F03 stages and checks the **initial** output in `apps/xInstGraph/xInstGraph.hpp`. After startup, `m_graph.outputPath()` points at the published file and library calls still save directly to it. The open `m_outputFd` records which inode the app owns, but it does not protect later path-based writes.
- `MagAOXApp::handleSetProperty()` invokes the registered callback and discards its return value (`libMagAOX/app/MagAOXApp.hpp:3555-3591`). Returning `-1` from `igHandleSetProperty()` alone will not stop the app after a publication failure.
- `instGraph` is a separate CMake library installed under `/usr/local`; MagAOX links `-linstGraph`. The library change must be built and installed before building the app against its new API. The library checkout currently has no dedicated test target. The normal `MagAOXApp` sequence calls `appStartup()` before `startINDI()` (`libMagAOX/app/MagAOXApp.hpp:1829-1861`).

## Publication contract

1. Keep all graph mutations for one INDI callback in memory. No library method may write the published path during that callback. Serialize the final graph once after every interested node has handled the message. Ignore unmatched property keys without publishing.
2. Write the serialized XML through an exclusively created temporary file descriptor in the destination directory. Check serialization, every write (including partial writes and `EINTR`), file size, permissions, and `fsync` before publication. A failed preparation leaves the current output byte-for-byte unchanged and removes the temporary file.
3. Before replacement, confirm the destination is still the regular file published by this run, using the retained device/inode identity. Recheck immediately before `rename`. Replace by same-directory `rename`, then transfer ownership to the new inode and close the old descriptor. `graph.clobberOutput=false` does not block replacement of this owned output.
4. If the destination has disappeared or was replaced, do not overwrite the new occupant. Treat that as a publication failure and leave it untouched. Use a separate owned-output check for updates: the startup `checkOutputPath()` rejects an existing output when `clobberOutput=false`, so it cannot be reused unchanged. Recheck that the destination does not alias the input before publication. Serialize callbacks and publication so two updates cannot interleave.
5. On any node-handler or publication failure, log the cause, return failure from the callback, and latch a fatal update error. Have `appLogic()` return failure when latched so the main loop performs shutdown; the base INDI dispatcher ignores callback return values. Do not publish a partially updated in-memory graph. The previous complete output remains available until the existing shutdown ownership cleanup runs.
6. This plan guarantees atomic **visibility** of each published snapshot and checked file preparation. It does not claim crash durability of the directory entry. An unrelated process can still race between identity check and `rename`; document that limitation and use an app-controlled output directory where possible.

## Library work: `/home/jrmales/Source/instGraph`

1. Add a documented way to disable automatic file saves on `instGraphXML`, defaulting to the current automatic behavior for other clients. In that mode, `stateChange()`, `valuePut()`, and `valueExtra()` still update the XML but never call `save_file()`. Keep propagation semantics unchanged.
2. Add a checked explicit serialization API, for example `serializeXML(std::string &xml, std::string &error) const`. Use pugixml's stream serialization and check the stream state and nonempty result; report allocation/serialization errors. Do not expose the internal `pugi::xml_document` to the app or modify vendored pugixml.
3. In the legacy automatic mode, centralize `save_file()` and check its boolean result. Because the existing virtual mutation methods return `void`, report a failed save with a documented exception rather than silently continuing. Confirm whether any library demos need to catch it.
4. Add Catch2 tests for disabled autosave, explicit serialization, and the checked legacy failure path using the mxlib conventions below. Build both shared and static targets. Keep the new public header and installed library in sync.

## instGraph Catch2 structure and documentation (mxlib conventions)

- Add a repository-local Catch2 v2.13.9 single header and its license, matching mxlib's `tests/catch2/catch.hpp`. `tests/testMain.cpp` defines `CATCH_CONFIG_MAIN`, and `tests/src/instGraphXML_test.cpp` mirrors `src/instGraphXML.cpp`. Declare Catch2 as a test-only dependency in CMake. Build the test source as its own executable, register it with CTest, and provide focused build/run commands.
- Put a top Doxygen `\file` and `\brief` block on each new test source, without an `\author` tag. Add `unit_tests` and `instGraphXML_unit_tests` groups in `doc/groupdefs.dox`; put the test file and every `TEST_CASE` in the leaf group. Add `tests/src` to `doc/instGraph.dox` input (its current `../tests` entry is commented out and recursion is disabled), and verify generated documentation includes the cases.
- Use `TEST_CASE` for every top-level test. Place a brief `///` immediately before each case whose text exactly matches its Catch2 name; follow it with a short `/** ... */` block containing `\ingroup instGraphXML_unit_tests` and any needed behavior detail. Use `SECTION` within a case for variants; do not use `SCENARIO` declarations.
- Keep `REQUIRE`, `CHECK`, and related assertions inside `TEST_CASE` or `SECTION` bodies. Helpers may create fixtures or return named observations. Use `CAPTURE` or `INFO` at assertions with loop or failure-injection context. Call the production API directly in test bodies so Doxygen's `Referenced by` links reach the real API. When a fixture or fault-injection wrapper hides a call, add a raw Doxygen-only reference inside the case under `#ifdef __DOXY_ONLY__` and hide harness helpers with `\cond`/`\endcond` as needed. Run `clang-format` on touched C++ files.

These conventions come from `/home/jrmales/Source/mxlib/AGENTS.md`, `doc/documenting.dox`, `tests/CMakeLists.txt`, and `tests/testMain.cpp`. The mxlib checkout has unrelated working-tree edits; it is a read-only style reference for this work.

## MagAOX work: `apps/xInstGraph`

1. Disable library autosave before node construction in `loadConfigImpl()`. Keep the F01/F03 staging and output path validation. Replace the startup `stateChange()` side-effect save with an explicit checked serialization into the already open stage descriptor, then publish using the existing no-replace or clobber policy.
2. Factor a helper that writes one serialized snapshot to an owned descriptor, handles short writes and `EINTR`, verifies a nonempty regular file, sets the intended mode, and calls `fsync`. Use it for both startup and callback publication. Keep the temporary path and descriptor under cleanup control on every error path.
3. In `igHandleSetProperty()`, hold an app mutex over node dispatch, final `stateChange()`/serialization, and publication. Create a fresh same-directory temporary file for each matched callback; check the currently published inode before replacing it. After successful `rename`, update `m_outputIdentity` and `m_outputFd` so shutdown removes only the latest file owned by this run. Mark the error latch on handler or publication failure; make `appLogic()` act on it.
4. Serialize callback publication with shutdown. Normal execution starts INDI only after `appStartup()` succeeds, so the initial snapshot is already published; retain a guard against direct test calls or any future earlier dispatch. Do not hold the graph mutex across an operation that could synchronously invoke a callback.
5. Update the README's output contract and the F01/F03 plan's execution notes if the startup staging behavior changes. Follow the MagAOX `AGENTS.md` documentation pass and `clang-format` rules for every touched C++ file. Keep library and app changes in their own repository branches and commits; install/test the matching library version before testing MagAOX.

## Acceptance tests

- One callback that changes a put, propagates a beam/link state, and changes an extra value yields one final snapshot with all expected fields. Observe the old complete snapshot while the new temporary file is being written, then a complete new snapshot after publication; parse both with `instGraphXML`.
- Use a deterministic test seam to inject serialization, write, `fsync`, and `rename` failures. Each failure is logged, makes `appLogic()` fail, leaves the previous output unchanged until shutdown, and leaves no temporary file. Separately inject partial writes and `EINTR` followed by successful writes to verify the write loop completes the snapshot. A subsequent callback cannot publish after the failure latch is set.
- Replace the published path with a different regular file or a symlink before a callback. The app refuses to overwrite it and does not remove it on shutdown. Repeat with an alias to the input and verify the input bytes are unchanged.
- Verify `graph.clobberOutput` still governs only initial replacement, startup still publishes before the first callback, and normal shutdown removes the latest owned output.
- Run the new Catch2 executable through CTest, verify its Doxygen group and production API links, rebuild/install the library in a controlled local environment, build `apps/xInstGraph`, and run the focused `xInstGraph_test` with the matching shared library (`LD_LIBRARY_PATH=/usr/local/lib` for the current setup). Check both repositories' diffs and worktrees before committing.

## Follow-up boundary

Failure after an atomic `rename` but before directory metadata reaches stable storage can lose the newest publication after a crash. If crash durability is required, add a directory `fsync` and specify how to report its post-publication failure. Fully excluding a concurrent external replacement at the exact `rename` boundary requires stronger control of the output directory than a device/inode precheck provides.

## Execution notes (2026-09-28)

- `instGraphXML` now supports disabling automatic saves and checked explicit XML serialization (instGraph commit `59a8adc` on `jrmales/atomic-graph-updates`). Legacy automatic saves throw on a failed `save_file`. A repository-local Catch2 v2.13.9 harness and source-mirrored `instGraphXML_test.cpp` run through CTest; generated Doxygen includes the unit-test group and direct API references.
- xInstGraph disables automatic saves before configuring nodes. Startup and each matched INDI callback write through an owned staging descriptor, check short writes, `EINTR`, file identity, mode, size, and `fsync`, then publish the complete snapshot. Callback updates replace only the inode owned by the app. A failure latches until `appLogic()` initiates shutdown.
- The library CMake build and CTest passed. The xInstGraph app built against the matching `/tmp` library install, and the focused app suite passed 9 cases with 142 assertions, including two successive successful updates and a callback with no matching node.
- Follow-up coverage support is in instGraph commits `764468f` and `ab37307`: the `coverage` target configures an isolated GCC build, captures a zero-count baseline, runs CTest, filters bundled dependencies, and embeds the `genhtml` report in Doxygen. The target passed with 1/1 tests and currently reports 301/1229 source lines covered (24.5%); rebuilding `docs` preserved the embedded report.
- The atomic visibility and external replacement race limits under the publication contract remain. The matching library must be installed before an operational xInstGraph build.

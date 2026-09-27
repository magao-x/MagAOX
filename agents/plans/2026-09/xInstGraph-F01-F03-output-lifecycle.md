# xInstGraph F01 and F03: initial publication and output ownership

## Objective

Address F01 and F03 in the [xInstGraph review](xInstGraph-review.md). A successfully started app should publish a complete draw.io output before its first INDI update. By default, an existing output is preserved. An explicit config option may replace an existing output. The output must never refer to the input diagram, regardless of that option.

## Current behavior

- loadConfigImpl() sets the library's final output path before loading and configuring nodes (apps/xInstGraph/xInstGraph.hpp:156-177). Node configuration may cause the library to save to that path before hideLinks() and hidePuts() run (:189-405).
- hideLinks() and hidePuts() change the in-memory XML but do not save it (instGraph/src/instGraphXML.cpp:999-1034). appStartup() registers callbacks and enters READY without a final publication (apps/xInstGraph/xInstGraph.hpp:440-499).
- The library's stateChange(), valuePut(), and valueExtra() save directly to the output path (instGraph/src/instGraphXML.cpp:833-996). The app removes that path unconditionally at shutdown (apps/xInstGraph/xInstGraph.hpp:506-511).
- The existing app test builds but fails its output-exists assertion because its fixture has no update that triggers a library save (apps/xInstGraph/tests/xInstGraph_test.cpp:46-159).

## Output policy

1. Require an explicit, nonempty graph.outputPath. Add graph.clobberOutput as a boolean config option, default false. With false, any existing destination causes a clear error and is neither overwritten nor removed. With true, an existing regular output file may be replaced at initial publication; symlinks and other file types remain rejected. Failures before publication leave an existing output intact.
2. Resolve both paths and reject input/output identity unconditionally, before any output write. Compare canonical/equivalent files when the destination exists, so relative paths, symlinks, hard links, and alternate directory spellings cannot bypass the rule. Reject a directory as output in both modes. Recheck identity immediately before publishing, after configuration has completed.
3. Send all configuration-time library writes to a unique, exclusively created staging file in the output directory. The final output path is not passed to instGraph until the initial graph is complete.
4. After callback registration succeeds, call m_graph.stateChange() once to write the final initial state to staging. With clobberOutput=false, publish with an atomic no-replace operation (for example, linkat followed by unlink of staging). With clobberOutput=true, publish with an atomic replacement operation after the alias recheck. In neither mode should an existing destination be truncated during preparation. Switch m_graph.outputPath() to the published path only after publication succeeds, then enter READY.
5. Record the identity of the file successfully published by this run. On normal shutdown or a later startup failure, remove only that owned output if the path still refers to the same file. In clobber mode the old file has been intentionally replaced; the new file is owned by this run and is removed on shutdown. Clean up an unpublished staging file on every error path.
6. Subsequent graph updates may write to the owned output as today. F02 will address their unchecked and non-atomic writes. A concurrent replacement of the output after publication remains an F02 write-path concern.

## Implementation steps

1. Add the config option in setupConfig() and load it in loadConfigImpl(). Add documented output-path, staging-path, and ownership state to xInstGraph. Keep helpers for path identity checks and staging/publishing if they make the lifecycle clearer.
2. In loadConfigImpl(), validate the input and destination paths, load the source graph, create staging in the destination directory, set the library output path to staging, then construct/configure nodes and hide links/puts. Use RAII or equivalent cleanup so exceptions cannot leave staging behind.
3. In appStartup(), after all INDI keys register and before setting READY, force a graph snapshot to staging and verify that it produced a nonempty file. Recheck input/output identity and apply the selected no-replace or replace publication operation. Record ownership only after successful publication.
4. Make appShutdown() remove the published output only when ownership and file identity still match; also clean up staging if startup did not publish.
5. Update test fixtures and documentation across each touched C++ file per AGENTS.md. Run clang-format on touched C++ files. Keep the combined functional change, tests, and plan in one feature-branch commit; separate unrelated documentation or formatting cleanup.

## Tests and acceptance criteria

- With no INDI callback, a successful appStartup() produces a parseable output file. The existing test should assert this before sending its first property.
- A fixture containing a node, input, output, and internal link produces an output whose link has opacity=0 and whose puts have opacity=0 and textOpacity=0 after startup. Configure a static state that causes an earlier staging save, so the test detects a missing final publication.
- With clobberOutput omitted or false, a pre-existing output file or symlink causes failure; its contents and directory entry remain unchanged after config, startup, and shutdown.
- With clobberOutput=true, successful appStartup() replaces a pre-existing regular output with the new graph, and shutdown removes the file published by this run. A failure before publication leaves the previous output intact. A later failure, such as INDI startup failure, can occur after the old output has been replaced.
- Setting output equal to input fails in both modes. Test exact path, relative spelling, symlink, and hard-link aliases, and verify the input content remains unchanged.
- A new output is removed on normal shutdown in either mode. If the published path is replaced with another file before shutdown, that replacement survives. A failed config or startup removes only its own staging file.
- Build and run xInstGraph_test with LD_LIBRARY_PATH=/usr/local/lib, and build the app. Use fresh per-test paths so a previous run cannot make an output-exists assertion pass.

## Remaining edge

The library still ignores save_file() failures and writes directly to its path on later state changes (F02). The initial publication should detect an empty or missing staging file, but reliable error reporting and atomic replacement for every later callback remain F02 work.

## Execution notes (2026-09-27)

- Implemented `graph.clobberOutput` with a false default. Configuration rejects an existing destination unless clobber is enabled, and clobber accepts only an existing regular file. An output that resolves to the input is rejected in either mode, including relative, symbolic-link, and hard-link aliases.
- Configuration-time graph saves go to an exclusive staging file in the output directory. After callback registration, startup saves the final hidden-link/hidden-put state, checks for a nonempty staging file, and publishes with `link`/`unlink` for no-replace or `rename` for clobber.
- The app records device and inode identities and retains an open descriptor for each owned file, then removes a stage or published output only when its path still names that inode. Failed configuration and startup clean their stages.
- Added isolated unit fixtures covering initial publication, final hidden state, both clobber settings, input aliases, symlink destinations, failed startup, and shutdown after an external replacement. The targeted test run passed all 5 cases and 61 assertions; the app build also passed.
- F02 remains: the instGraph library's later `save_file()` calls still write directly to the published path without checking write errors or replacing atomically.

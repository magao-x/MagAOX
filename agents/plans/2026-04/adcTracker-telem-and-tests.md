Task: We need to add several features to the adcTracker app and get it's testing coverage up.  The most important feature is to add telemetry for the properties which can be modified online by INDI.  We also need to implement tests of the appStartup and appLogic behavior.  We are on the jrmales/tracker-crash-guards feature branch which already contains updates to harden the code.  We'll continue working on this branch.

Review AGENTS.md. Then please formulate a plan and describe it below under `Plan:`.  Do not modify this prompt.   

Plan:
1. Review the current `adcTracker` implementation, nearby tracker/telemeter patterns, and the existing test harness so the work follows local MagAO-X app, telemetry, and Doxygen conventions without broad refactors.
   This includes confirming how other tracker-style apps record online-adjustable state, how MagAO-X logger schemas are registered, and what testing seams are available without introducing a large production refactor.

2. Add tracker-specific telemetry for the values that can be changed while the app is running through INDI.
   The telemetry should focus on the operator-adjustable state, not the static config loaded once at startup.
   The planned telemetry payload is:
   - tracking enabled/disabled state
   - `deltaAngle`
   - `adc1delta`
   - `adc2delta`
   - `minZD`
   These are the values currently exposed for live adjustment through INDI and therefore the values most important to preserve in the telemetry stream.
   The current plan is to add a dedicated `telem_adctrack` logger type rather than overloading an existing telemetry record, because `telem_position` is too narrow and the payload is materially different from the existing HWP/K tracker telemetry.

3. Integrate `dev::telemeter<adcTracker>` into the app in a minimal, local way.
   The implementation should:
   - add the telemeter base/helper typedef/friend wiring in `adcTracker`
   - initialize telemeter config and startup/shutdown hooks
   - provide `checkRecordTimes()` and `recordTelem(...)` for the ADC tracker telemetry type
   - record an initial telemetry snapshot once startup succeeds
   - refresh telemetry when the online-adjustable values change in the INDI callbacks
   - continue servicing periodic telemeter logic from `appLogic()` without changing the existing ADC target calculation behavior
   This should preserve existing behavior while making online changes visible in the telemetry logs.

4. Add the supporting logger plumbing required by the new telemetry record.
   The expected touch points are:
   - `libMagAOX/logger/types/schemas/telem_adctrack.fbs`
   - `libMagAOX/logger/types/telem_adctrack.hpp`
   - `libMagAOX/logger/logCodes.dat`
   - `libMagAOX/logger/generated/logTypes.hpp` and related generated logger artifacts after regeneration
   - `libMagAOX/logger/types/telem.cpp`
   - generated flatbuffer headers / schema binaries / generated logger tests as needed by the normal logger build flow
   We should choose the next available telemetry event code near the tracker telemetry range and let the normal logger generation step rebuild the generated files.

5. Expand `apps/adcTracker/tests/adcTracker_test.cpp` to cover `appStartup()` and representative `appLogic()` behavior, while preserving the callback validation tests and bringing the file documentation up to current style.
   The intended test coverage is:
   - configuration loading leaves documented defaults unchanged when no overrides are provided
   - configuration loading applies explicit overrides for the ADC tracker parameters
   - existing INDI callback validation coverage stays in place
   - direct callback behavior tests verify that valid INDI payloads actually update the corresponding runtime state
   - `appStartup()` succeeds with a valid lookup table and leaves the tracker ready to run
   - `appStartup()` fails when the lookup table cannot be read
   - `appStartup()` fails on malformed lookup-table content such as too few rows, inconsistent column sizes, non-finite values, or non-monotonic zenith-distance samples
   - `appStartup()` catches both `std::exception` and non-standard exceptions raised while constructing the interpolators
   - `appLogic()` does nothing when tracking is disabled
   - `appLogic()` does nothing before valid ZD is received
   - `appLogic()` computes and dispatches the expected ADC commands when tracking is enabled and the update interval has elapsed
   - `appLogic()` uses zero offsets below `minZD`
   - `appLogic()` clamps to the last lookup-table row above `maxZD`
   - `appLogic()` catches both `std::exception` and non-standard exceptions raised during interpolation
   - the `teldata` callback catches both `std::exception` and non-standard exceptions raised while extracting `zd`
   - `appShutdown()` is exercised directly
   - `recordTelem(...)` is exercised directly
   - the remaining branch coverage should come from direct static callback-wrapper calls, explicit missing-element / invalid-value callback payloads, and per-axis ADC send helpers rather than from adding another outbound-command indirection layer
   To keep the production change small, the tests will likely use a narrow seam in the test harness to capture the computed outbound ADC targets instead of requiring a live INDI connection.
   The current preference is to use a few protected virtual seams in the app test harness for exception injection rather than header multi-include `#define` fault macros, since `adcTracker` is an app class and the seam approach is more local and easier to maintain here.

6. Update test documentation and grouping for the touched test file.
   This should include:
   - a stronger file header in `apps/adcTracker/tests/adcTracker_test.cpp`
   - Doxygen blocks for each test scenario / case
   - adding the app test under `app_unit_test` in `tests/groups.dox`
   - using `tests/testXWC.hpp` and `XWCTEST_DOXYGEN_REF(...)` if needed so Doxygen still links the real `adcTracker` API under test despite the harness/subclass indirection

7. Run the required cleanup and verification steps after the implementation is in place.
   The plan is to:
   - run `clang-format` on touched source/header/test/logger files
   - regenerate logger artifacts through the normal build path if the new telemetry schema is added
   - build and run the relevant `adcTracker` unit test target, plus any logger-generated test target that needs to be refreshed
   - note any residual edge cases, especially around malformed lookup tables and telemetry/log generation dependencies

Execution notes:
- Startup logging now records a normal INFO-style message after the lookup table is successfully read, including the resolved path, row count, and range covered by the table, instead of printing a transient `std::cerr` message before parsing.
- The runtime status surface now includes two read-only INDI switch properties, `belowMinZD` and `aboveMaxZD`, which track whether the latest valid ZD is currently below `minZD` or above `maxZD`.
- Threshold-crossing behavior should emit a WARNING exactly once each time the tracker enters the below-minZD or above-maxZD state, while clearing the status switches silently when the ZD returns in range or becomes unavailable.

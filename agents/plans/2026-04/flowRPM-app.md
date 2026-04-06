Original write-up:

> We need a new app called "flowRPM" that reads fan RPM from a file (which is updated at intervals by a systemd unit) and converts it to LPM.  The file is by default located at `/tmp/fac_flow.txt` and has the format:
> ```
> 1775430287 145131374
> 36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'
> ```
> where the first line contains a timespec and we want to extract the Fan RPM (1900.00 here). The RPM is mL/min, so in this case the report value should be 1.9 LPM.
>
> If the timespec is greater than a set time (default 60 sec), the file is not available for that time, or it fails to parse for that time, then the app should:
>  - log an error with back off in the reporting frequency to 1/60 sec
>  - report a bad value sentinel (-999 is typical)
>
> Parsing errors should include not finding the fan descriptor, the units not being RPM, and it now sayin 'OK'.
>
> We should have the following configurables:
>  - path to file
>  - time to wait to declare an error
>  - the fan descriptor, in this case `CHA_FAN1`, which is used to detect a wrong fan.
>
> We should have the following RO INDI property:
>  - status.flow_rate (updates on a new value)
>  - status.age (updates once per appLogic with the age of the displayed value)
>
> The app should be a telemeter.
>
> Follow koolanceCtrl and usbtempMon for examples of similar apps.
>
> We should have 100% statement and function test coverage following the CATCH2 framework and standards used in the project.
>
> Please review AGENTS.md then design this app and document the design and plan below in this file:
>
> Plan:

Derived planning document:

Prompt: We need a new app called `flowRPM` that reads fan RPM from a file written by a systemd unit, converts the reported value to LPM, exposes the result through INDI, and records telemetry. Review `AGENTS.md`, design the app, and document the plan in this file.

## Input Summary

The source file is expected by default at `/tmp/fac_flow.txt` with contents like:

```text
1775430287 145131374
36 | CHA_FAN1         | Fan          | 1900.00    | RPM   | 'OK'
```

Interpretation:

- line 1 is a source timestamp in `tv_sec tv_nsec` form
- line 2 is a pipe-delimited sensor record
- the value field `1900.00` is treated as `mL/min`
- the app reports `1.9` as `LPM`

Requested behavior:

- configurable file path
- configurable staleness timeout, default `60` seconds
- configurable fan descriptor, default `CHA_FAN1`
- RO INDI elements:
  - `status.flow_rate`
  - `status.age`
- telemetry support through `dev::telemeter`
- parse / stale / unavailable failures should:
  - log errors with rate limiting
  - publish a bad-value sentinel, typically `-999`
- unit tests should provide full statement and function coverage for the app-specific logic

## Clarified Design Decisions

The original prompt leaves a few points ambiguous. This plan resolves them as follows so implementation can proceed cleanly:

- "back off in the reporting frequency to `1/60 sec`" is interpreted as "at most once every 60 seconds"
  - `1/60 sec` would increase log frequency, so the implementation should instead rate-limit repeated error logs to one per 60 seconds while the fault persists
- the parser should validate that the units token in the file is literally `RPM`
  - even though the numeric value is then converted as though it represents `mL/min`
  - this preserves the explicit requested parse check while following the requested conversion behavior
- the status token requirement is interpreted as requiring the literal token `'OK'`
  - any other status string is a parse failure
- `status.flow_rate` and `status.age` should be implemented as two elements of a single RO INDI number property named `status`
- `status.age` should be computed from the timestamp in line 1, not from filesystem metadata
- recovery from an error should be immediate on the first successful fresh parse
  - publish the recovered value
  - update `status.age`
  - resume normal change-driven / telemeter-driven reporting cadence

## Proposed App Shape

Create a new app under:

- `apps/flowRPM/Makefile`
- `apps/flowRPM/flowRPM.cpp`
- `apps/flowRPM/flowRPM.hpp`
- `apps/flowRPM/tests/flowRPM_test.cpp`

The app should follow the `koolanceCtrl` and `usbtempMon` structural pattern, but it does not need `tty::usbDevice`. The class should instead derive from:

- `MagAOXApp<true>`
- `dev::telemeter<flowRPM>`

Recommended class-level structure:

- `friend class flowRPM_test;`
- `friend class dev::telemeter<flowRPM>;`
- `typedef dev::telemeter<flowRPM> telemeterT;`
- standard lifecycle methods:
  - `setupConfig()`
  - `loadConfig()`
  - `appStartup()`
  - `appLogic()`
  - `appShutdown()`
- a testable config-loading helper:
  - `int loadConfigImpl( mx::app::appConfigurator & _config );`
- a testable parse helper kept out of the class body:
  - read file text
  - parse timestamp and sensor line
  - validate descriptor / units / status
  - convert to `LPM`
  - report age / sentinel / error kind

## Internal State and Configuration

### Configurable parameters

Add app-specific configuration entries such as:

- `input.path`
  - default `/tmp/fac_flow.txt`
- `input.maxAge`
  - default `60`
  - the maximum acceptable age in seconds before the value is declared stale
- `input.fanDescriptor`
  - default `CHA_FAN1`
- `input.badValue`
  - default `-999`
  - optional, but useful for testability and consistency with other apps
- `input.errorLogInterval`
  - default `60`
  - rate limit for repeated fault logs

Also integrate telemeter configuration using the helper macros required by `AGENTS.md` rule 18:

- `TELEMETER_SETUP_CONFIG(config)`
- `TELEMETER_LOAD_CONFIG(_config)`
- `TELEMETER_APP_STARTUP`
- `TELEMETER_APP_LOGIC`
- `TELEMETER_APP_SHUTDOWN`

### Suggested member state

Recommended members:

- `std::string m_inputPath`
- `double m_maxAge`
- `std::string m_fanDescriptor`
- `double m_badValue`
- `double m_errorLogInterval`
- `double m_flowRate`
- `double m_lastLoggedFlowRate`
- `double m_age`
- `timespec m_sourceTs`
- `bool m_haveValidReading`
- `timespec m_lastErrorLogTs`
- `std::string m_lastErrorKey`
- `pcf::IndiProperty m_indiP_status`

Purpose:

- `m_flowRate` and `m_age` hold the currently published value
- `m_haveValidReading` separates "last value was valid" from "currently publishing sentinel"
- `m_lastLoggedFlowRate` supports change-driven telemetry / INDI updates without spamming
- `m_lastErrorLogTs` and `m_lastErrorKey` implement fault-log rate limiting

## Parsing and Validation Design

Split the app-specific logic so tests can cover it without depending on the full MagAO-X runtime:

1. File read step
   - try to open and read the file
   - failure is a distinct error condition

2. Timestamp parse step
   - parse line 1 into `tv_sec` and `tv_nsec`
   - reject missing line 1
   - reject malformed integer fields
   - reject `tv_nsec < 0` or `tv_nsec >= 1000000000`

3. Sensor line parse step
   - parse line 2 as pipe-delimited fields
   - trim whitespace around tokens
   - require enough columns to read:
     - channel index
     - descriptor
     - type
     - numeric value
     - units
     - status

4. Semantic validation
   - descriptor must equal configured `m_fanDescriptor`
   - units must equal `RPM`
   - status must equal `'OK'`
   - numeric field must parse as `double`

5. Conversion
   - compute `flow_lpm = parsed_value / 1000.0`

6. Age calculation
   - compare source timestamp to current `CLOCK_REALTIME`
   - if computed age is negative because of clock skew or race, clamp to `0`
   - if `age > m_maxAge`, treat as stale

7. Error publication
   - set `m_flowRate = m_badValue`
   - continue publishing `m_age` from the stale timestamp when parse succeeded but freshness failed
   - for missing file or unparsed timestamp, set `m_age = m_badValue`

Recommended explicit error categories:

- file open/read failure
- missing first line
- malformed timestamp
- missing second line
- malformed delimited record
- wrong descriptor
- wrong units
- bad status
- bad numeric value
- stale reading

This gives the app deterministic logging and makes test coverage straightforward.

## INDI Design

Expose one RO number property:

- property name: `status`
- elements:
  - `flow_rate`
  - `age`

Startup behavior:

- create both elements in `appStartup()`
- initialize both to the configured bad-value sentinel
- register as RO

Update behavior:

- on successful parse:
  - publish `flow_rate = converted LPM`
  - publish `age = current age in seconds`
- on failure:
  - publish `flow_rate = m_badValue`
  - publish `age` as:
    - computed stale age for stale-data cases
    - `m_badValue` for missing-file / malformed-timestamp cases

Publish cadence:

- `status.age` updates every `appLogic()` pass as requested
- `status.flow_rate` updates whenever the underlying displayed value changes
- if the app remains in a faulted state with unchanged sentinel, avoid redundant property writes unless the framework already expects them

## Telemetry Design

The app should participate in `dev::telemeter` like the existing telemeter-backed monitoring apps.

Implementation pattern:

- define a dedicated telemetry type, likely `telem_flowrpm`, to hold:
  - `flowRate`
  - `age`
  - an optional validity flag or sentinel-based convention

Preferred logger shape:

- a new logger type is better than reusing an unrelated telemetry record
- a validity bit is cleaner than inferring invalidity from `-999`, but if existing logger conventions strongly prefer sentinel-only numeric payloads, the record may omit the boolean and rely on the sentinel

App interface additions:

- `int checkRecordTimes();`
- `int recordTelem( const telem_flowrpm * );`
- `int recordFlow( bool force = false );`

Behavior:

- `checkRecordTimes()` returns `telemeterT::checkRecordTimes( telem_flowrpm() );`
- `recordTelem( const telem_flowrpm * )` delegates to `recordFlow( true )`
- `recordFlow( bool force )` logs whenever:
  - `force == true`
  - `m_flowRate` changed
  - validity changed
  - an operator-relevant change in the displayed state occurred

This gives immediate records on transitions plus periodic records through the telemeter max interval.

## FSM / Runtime Behavior

The app can stay structurally simple:

1. `setupConfig()`
   - add app config keys
   - call `TELEMETER_SETUP_CONFIG(config)`

2. `loadConfigImpl(...)`
   - load defaults and overrides
   - call `TELEMETER_LOAD_CONFIG(_config)`

3. `appStartup()`
   - create and register the RO INDI property
   - initialize to sentinel values
   - call `TELEMETER_APP_STARTUP`
   - set the app state to `READY`

4. `appLogic()`
   - read and parse the current file contents
   - update displayed values and log any newly entered fault state
   - update `status.age` every loop
   - call `recordFlow()` when the displayed reading changes
   - call `TELEMETER_APP_LOGIC`
   - remain in `READY` unless telemeter startup fails or another fatal framework error occurs

5. `appShutdown()`
   - call `TELEMETER_APP_SHUTDOWN`

Because the data source is a passive file, parse failures should normally not drive the app into `FAILURE` or `NODEVICE`. The app should continue running, publish the sentinel, and recover automatically when the file becomes healthy again.

## Logging and Fault-Handling Policy

Log failures as operator-actionable software errors, but rate-limit repeated messages.

Recommended policy:

- log immediately on entering a new error condition
- suppress repeats of the same error key until `m_errorLogInterval` seconds have passed
- log recovery once when transitioning from invalid to valid

Suggested error keys:

- `open_failed`
- `timestamp_parse_failed`
- `record_parse_failed`
- `wrong_descriptor`
- `wrong_units`
- `bad_status`
- `stale`

This keeps logs useful when the systemd writer is absent or broken for a long time.

## Unit-Test Plan

The reference app tests are mostly placeholders, so this app should introduce real coverage around the app-specific logic.

Primary target:

- `apps/flowRPM/tests/flowRPM_test.cpp`

Test at least the following cases:

1. Configuration defaults load correctly.
   - default file path
   - default age limit
   - default descriptor
   - default sentinel
   - telemeter config path is exercised through the app config helper

2. Happy-path parse succeeds.
   - valid timestamp
   - matching descriptor
   - `RPM` units
   - `'OK'` status
   - numeric conversion `1900.00 -> 1.9`

3. Descriptor mismatch fails.

4. Units mismatch fails.

5. Status mismatch fails.

6. Missing file produces sentinel and logs the proper error class.

7. Missing first line fails.

8. Malformed timestamp fails.

9. Missing second line fails.

10. Malformed numeric value fails.

11. Stale timestamp produces sentinel and preserves meaningful age.

12. Recovery from failure publishes the valid value again.

13. Repeated identical failure does not request a fresh log before the backoff interval expires.

14. `recordTelem(...)` forces a telemetry record.

15. `checkRecordTimes()` wires to the `telemeter` base correctly.

To make full statement and function coverage realistic, keep the parsing and error-decision logic in small helpers that can be called directly by the test harness.

## Build and Integration Tasks

1. Add the new app directory and target makefile entry.
2. Add the new test binary to the relevant test lists/build plumbing.
3. Add the new logger type and schema if telemetry requires a new message type.
4. Confirm the app handbook link and documentation group names follow the standard MagAO-X app pattern.
5. Run `clang-format -i` on touched code files.

## Implementation Plan

1. Scaffold `apps/flowRPM` from the standard lightweight monitoring-app pattern.
   - add `flowRPM.cpp`
   - add `flowRPM.hpp`
   - add `Makefile`
   - add the tests file

2. Implement the class declaration and lifecycle wiring.
   - inherit from `MagAOXApp<true>` and `dev::telemeter<flowRPM>`
   - add the telemeter typedef, friend declarations, and interface methods
   - keep non-trivial definitions out of the class declaration

3. Add configuration support and the RO INDI property.
   - `input.path`
   - `input.maxAge`
   - `input.fanDescriptor`
   - `input.badValue`
   - `input.errorLogInterval`
   - `status.flow_rate`
   - `status.age`

4. Implement a small parse/result layer that is easy to unit test.
   - a parsed reading structure
   - a parse result enum or error key
   - helpers for trimming, splitting, timestamp parsing, and conversion

5. Implement runtime publication behavior.
   - publish valid values on success
   - publish sentinel on failure
   - update age every loop
   - log errors with backoff
   - log recovery once

6. Add telemetry support.
   - add or select the telemetry record type
   - implement `checkRecordTimes()`
   - implement `recordTelem(...)`
   - implement `recordFlow(bool force)`

7. Add thorough Catch2 coverage for parser, state transitions, and telemeter hooks.

8. Format and verify.
   - `clang-format -i` on touched source/header/test files
   - build the new app and its test target
   - run the targeted tests

9. Keep the work cleanly separated if this moves to implementation.
   - functional code change commit first
   - documentation commit second
   - formatting-only commit last if needed

## Notes / Expected Edge Cases

- The systemd writer may update the file non-atomically.
  - if partial reads are possible, parse failures should be treated as transient and recovered on the next successful cycle
- If the source timestamp is in the future, clamp age to `0` instead of faulting immediately
- If the descriptor text includes surrounding whitespace, trim before comparison
- If the file later contains multiple sensor rows, the parser should either:
  - explicitly require exactly one sensor row, or
  - scan rows until the configured descriptor is found

Preferred initial implementation:

- support the current two-line format exactly
- optionally allow scanning all data rows for the configured descriptor if that can be done without complicating tests too much

The simplest robust version is to support "timestamp line plus one or more sensor lines" and select the row whose descriptor matches `m_fanDescriptor`.

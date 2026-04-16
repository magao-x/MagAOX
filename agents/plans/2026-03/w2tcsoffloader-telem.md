Problem Statement: we need to add `dev::telemeter` functionality to the app `w2tcsOffloader` to record the calculated Zernike basis coefficients. Review `AGENTS.md` and document a plan in this document.

Comments on initial plan:
- be sure to use the TELEMETER macros for the interface calls in step 5.
- let's make the functional changes first, and have a commit on the new branch, before doing the documentation and a commit, and yet another commit for a formatting cleanup.  This should be the standard practice added to AGENTS.md to keep the changes cleanly separated.


Plan

1. Review and mirror the existing telemetry integration pattern used elsewhere in MagAO-X.
   - Use apps such as `usbtempMon`, `cacaoInterface`, and `t2wOffloader` as the implementation model for:
     - inheritance from `dev::telemeter<derivedT>`
     - the `telemeterT` typedef
     - `checkRecordTimes()`
     - `recordTelem(...)`
     - calling telemeter `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, and `appShutdown`
   - Keep changes scoped to `w2tcsOffloader` and the new logger type needed for the coefficient vector.

2. Add a purpose-built telemetry log type for the coefficient vector.
   - The existing `logger::telem_offloading` log type is not suitable because it only records:
     - `num_modes`
     - `num_average`
     - `fps`
   - The requirement here is to record the calculated Zernike basis coefficients themselves, so the best fit is a new flatbuffer-backed logger type modeled after `logger::telem_dmmodes`.
   - Proposed contents of the new log type:
     - a `vector<float>` of the offloaded coefficients
   - Proposed location:
     - `libMagAOX/logger/types/telem_w2tcsoffloader.hpp`
     - plus the corresponding flatbuffer schema under `libMagAOX/logger/types/schemas/`
   - Also update the logger registration/build plumbing consistently with existing logger types:
     - `libMagAOX/logger/types/telem.cpp`
     - `libMagAOX/logger/logCodes.dat`
     - any generated/header include lists if required by the build

3. Integrate `dev::telemeter` into `w2tcsOffloader`.
   - Update the class declaration in `apps/w2tcsOffloader/w2tcsOffloader.hpp` to:
     - inherit from `dev::telemeter<w2tcsOffloader>`
     - add `friend class dev::telemeter<w2tcsOffloader>;`
     - add `typedef dev::telemeter<w2tcsOffloader> telemeterT;`
   - Add the standard telemeter interface declarations:
     - `int checkRecordTimes();`
     - `int recordTelem( const telem_w2tcsoffloader * );`
     - `int recordZCoeffs( bool force = false );`
   - Add any member state needed to suppress redundant telemetry records, for example:
     - `std::vector<realT> m_zCoeffs;` as the current coefficient values
     - `std::vector<realT> m_lastZCoeffs;` as the last logged values

4. Make coefficient computation update internal state first, then publish both INDI and telemetry from that state.
   - In `processImage(...)`, compute coefficients into `m_zCoeffs` instead of writing only to the INDI property.
   - Explicitly zero coefficients at indices `>= m_nModes` so telemetry reflects the same effective command state sent to TCS.
   - Continue updating `m_indiP_zCoeffs` from the same `m_zCoeffs` values so INDI and telemetry remain consistent.
   - After updating the coefficients, call `recordZCoeffs()` so changes are logged promptly instead of waiting only for the max telemeter interval.
   - During startup after loading the mode cube:
     - clamp configured `m_nModes` to the number of available modes in the FITS cube
     - fail with a `LOG_CRITICAL` `text_log` and shutdown if the available mode count exceeds what can be represented by the two-digit INDI element naming scheme (`00` through `99`)

5. Wire the app lifecycle to the telemeter base class.
   - `setupConfig()`
     - use `TELEMETER_SETUP_CONFIG(config)` in addition to the existing shmim monitor setup
   - `loadConfigImpl(...)`
     - use `TELEMETER_LOAD_CONFIG(_config)`
   - `appStartup()`
     - use `TELEMETER_APP_STARTUP`
   - `appLogic()`
     - use `TELEMETER_APP_LOGIC` in operating states where telemetry logging is valid
   - `appShutdown()`
     - use `TELEMETER_APP_SHUTDOWN`

6. Implement the telemetry recording methods in the standard MagAO-X style.
   - `checkRecordTimes()`
     - return `telemeterT::checkRecordTimes( telem_w2tcsoffloader() );`
   - `recordTelem( const telem_w2tcsoffloader * )`
     - delegate to `recordZCoeffs(true)`
   - `recordZCoeffs( bool force )`
     - compare current coefficients against the last logged coefficients
     - log on any change or when `force == true`
     - emit a telemetry message containing the coefficient vector

7. Do a documentation/style pass across the touched files.
   - Keep file-level Doxygen blocks present and aligned with the local style.
   - Add `///` summaries and inline parameter docs for any new declarations in headers.
   - Keep declaration grouping stable and avoid introducing non-trivial in-class definitions.
   - If a lock-lifetime-only scope is introduced, annotate it as `{ //mutex scope`.

8. Run formatting and targeted verification.
   - Run `clang-format -i` on the touched source/header files.
   - Build the affected target(s) to catch schema/logger integration issues.
   - If available, confirm at runtime or in logs that:
     - telemetry records are produced
     - coefficient vectors match the INDI values
     - repeated identical frames do not spam logs except at the configured maximum interval

9. Keep the work separated into clean commits on the feature branch.
   - Make the functional telemetry changes first and commit them on the new feature branch.
   - Follow with a documentation-only pass and commit.
   - If needed, make formatting-only cleanup a separate final commit.
   - Treat this functional/docs/formatting separation as the preferred standard workflow for similar changes going forward.

Notes / Expected Design Decisions

- Preferred logger shape:
  - a new vector-valued telemetry type, rather than extending `telem_offloading`
  - rationale: `telem_offloading` is already used for scalar offloading summary data in other apps, and overloading its meaning would make downstream interpretation ambiguous

- Recording behavior:
  - immediate record on coefficient change from `processImage(...)`
  - periodic forced record via `dev::telemeter` max-interval logic
  - rationale: this matches common MagAO-X patterns and avoids losing long steady-state periods in telemetry

- Scope:
  - this plan assumes no change to the offload math itself
  - the work is limited to exposing the already computed coefficient vector through telemetry
  - startup validation now also includes guarding the configured/available mode counts so the INDI naming scheme remains valid

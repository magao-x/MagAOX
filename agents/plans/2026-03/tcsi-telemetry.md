Prompt: add telemetry to `tcsInterface` for the offloading control parameters `m_offlTT_enabled`, `m_offlTT_gain`, `m_offlTT_thresh`, and the corresponding focus parameters `m_offlF_enabled`, `m_offlF_gain`, `m_offlF_thresh`. Also add `labMode` as separate telemetry. Work on branch `jrmales/tcsi-telem-always`.

## Analysis

The `apps/tcsInterface` app already maintains all of the requested state in memory and exposes it through INDI properties:

- `m_labMode`
- `m_offlTT_enabled`
- `m_offlTT_gain`
- `m_offlTT_thresh`
- `m_offlF_enabled`
- `m_offlF_gain`
- `m_offlF_thresh`

What is missing is telemetry plumbing for those values.

The existing `tcsInterface` telemeter interface only covers:

- `telem_telpos`
- `telem_teldata`
- `telem_telvane`
- `telem_telenv`
- `telem_telcat`
- `telem_telsee`

So this work is not just a small change in `tcsInterface`. It also requires adding one or more logger telemetry types in `libMagAOX`, wiring them into the telemeter cadence, and emitting records when values change.

## Design Choice

The request calls for:

- separate tip/tilt control telemetry
- separate focus control telemetry
- `labMode` as a separate telemetry record

The implementation should therefore use three telemetry log types:

1. one for tip/tilt offload control state
2. one for focus offload control state
3. one dedicated to `labMode`

Recommended names:

- `telem_tcsi_tiptilt`
- `telem_tcsi_focus`
- `telem_tcsi_labmode`

Tip/tilt and focus should remain distinct log types because they are operationally different controls and are used differently downstream.

To minimize duplicated logger code, the tip/tilt and focus log types can share:

- the same underlying flatbuffer schema shape
- the same helper or base-log implementation pattern

This should follow the same general idea as paired logger types such as `text_log` and `user_log`: separate log identities and event codes, but shared structure where practical.

`labMode` should remain its own log type because it is a separate application status rather than an offload-control parameter.

## Scope

### 1. Add logger types in `libMagAOX`

Create new telemetry definitions for:

- `telem_tcsi_tiptilt`
- `telem_tcsi_focus`
- `telem_tcsi_labmode`

For each new telemetry type:

- add a flatbuffer schema under `libMagAOX/logger/types/schemas`
- add the corresponding logger type header under `libMagAOX/logger/types`
- add the header to `libMagAOX/Makefile`
- add a code entry to `libMagAOX/logger/logCodes.dat`
- add a `lastRecord` initialization entry to `libMagAOX/logger/types/telem.cpp`

The tip/tilt record should include:

- `enabled`
- `gain`
- `thresh`

The focus record should include:

- `enabled`
- `gain`
- `thresh`

The tip/tilt and focus payloads should share the same schema shape so the two log types can reuse the same underlying field layout and basic logger implementation.

The lab-mode record should include:

- `labMode`

### 2. Extend `tcsInterface` telemeter declarations

In `apps/tcsInterface/tcsInterface.hpp`:

- add `recordTelem( const telem_tcsi_tiptilt * )`
- add `recordTelem( const telem_tcsi_focus * )`
- add `recordTelem( const telem_tcsi_labmode * )`
- add helper functions to write the records, e.g.:
  - `recordTcsiTipTilt( bool force = false )`
  - `recordTcsiFocus( bool force = false )`
  - `recordTcsiLabMode( bool force = false )`

Update `checkRecordTimes()` so the new telemetry types participate in the normal telemetry cadence.

This is important because MagAO-X telemetry should still be written on the configured interval even when values do not change.

### 3. Emit telemetry on change

Implement the new `record...()` helpers with the same pattern as the existing `recordTel...()` methods:

- keep static last-recorded values
- emit a telemetry record when any tracked value changes
- also emit when `force == true`

For tip/tilt control, track:

- `m_offlTT_enabled`
- `m_offlTT_gain`
- `m_offlTT_thresh`

For focus control, track:

- `m_offlF_enabled`
- `m_offlF_gain`
- `m_offlF_thresh`

For lab mode, track:

- `m_labMode`

### 4. Call record helpers at the right mutation points

After the helper methods exist, add record calls where these values can change:

- after config load establishes initial values
- in the `m_indiP_labMode` callback
- in the `m_indiP_offlTTenable` callback
- in the `m_indiP_offlTTgain` callback
- in the `m_indiP_offlTTthresh` callback
- in the `m_indiP_offlFenable` callback
- in the `m_indiP_offlFgain` callback
- in the `m_indiP_offlFthresh` callback

This ensures:

- immediate telemetry on user-visible control changes
- periodic telemetry through the telemeter framework

No additional record calls are needed for:

- `m_offlTT_dump`
- `m_offlTT_avgInt`
- `m_offlF_dump`
- `m_offlF_avgInt`

unless the requirements are later expanded.

### 5. Keep changes minimal and style-consistent

While touching `tcsInterface.hpp`:

- preserve existing behavior
- keep declaration ordering stable
- add any required Doxygen comments to new declarations
- follow existing naming conventions

For the new logger headers:

- include standard top-of-file Doxygen blocks
- match existing logger type structure and accessor style

## Verification Plan

### 1. Build verification

Rebuild the logger and app code paths affected by the new telemetry:

- `libMagAOX`
- `apps/tcsInterface`

This confirms:

- schema generation is correct
- new log-code registration is valid
- the new telemetry types are visible to the app

### 2. Test coverage

Update or add tests in `apps/tcsInterface/tests/tcsInterface_test.cpp` to cover at least:

- `labMode` callback changes the internal state and can trigger telemetry recording path
- offload TT enable/gain/thresh callbacks change the internal state cleanly
- offload F enable/gain/thresh callbacks change the internal state cleanly

If tip/tilt and focus are factored through a shared schema or common helper/base log, include at least a compile-level check that both `telem_tcsi_tiptilt` and `telem_tcsi_focus` instantiate and expose the correct accessors independently.

If direct log-capture tests are too heavy for the current test harness, document that limitation and at minimum verify the record-helper methods compile and are callable through the telemeter interface.

### 3. Runtime/manual verification

Manual verification should confirm:

1. `labMode` writes a telemetry record on startup and when toggled.
2. TT offload telemetry writes on startup and whenever enable/gain/thresh changes.
3. Focus offload telemetry writes on startup and whenever enable/gain/thresh changes.
4. The telemeter interval still forces records even when values remain unchanged.

## Execution Order

1. Create branch `jrmales/tcsi-telem-always` if not already active.
2. Add `libMagAOX` logger schemas and type headers.
3. Register the new telemetry types in `logCodes.dat`, `telem.cpp`, and `libMagAOX/Makefile`.
4. Extend `tcsInterface` telemeter declarations and implementations.
5. Add change-triggered record calls in the relevant callbacks and startup/config paths.
6. Update tests.
7. Build and verify.

## Risks and Notes

- The main ambiguity is naming of the new telemetry types. The recommendation above uses `telem_tcsi_tiptilt`, `telem_tcsi_focus`, and `telem_tcsi_labmode` for clarity and to avoid colliding with existing generic telemetry names.
- If the logger framework does not support literal inheritance between log-type structs cleanly, the practical fallback is to share the schema and duplicate only the thin type wrappers and event-code declarations.
- Because `tcsInterface` is implemented largely in the header, the documentation pass should cover the full touched regions in that file, not only the inserted lines.
- This plan intentionally keeps telemetry limited to the values explicitly requested and does not broaden scope to dump or averaging controls.

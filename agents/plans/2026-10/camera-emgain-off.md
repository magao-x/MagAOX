# Standard camera EM-gain-off request

## Request

Address [MagAOX issue #105](https://github.com/magao-x/MagAOX/issues/105):
"adding a standard request switch for any camera that exposes EM gain that
causes the EMG to be set to 1 or off (it's a camera specific callback so the
camera decides what it means)". The issue also requests a GUI button.

Before starting this change, committed the preceding temperature protection
(`7fd7db79`), reconfiguration fix (`100a0ac6`), and documentation cleanup
(`6c4f745c`) on `jrmales/camera-emgain-temperature`. The companion configuration
was committed as `0269288` on `jrmales/camera-temperature-cautions` in
`/opt/MagAOX/config`. No remote updates were made.

## Interface

- Every `stdCamera` with `c_stdCamera_emGain=true` exposes the standard request
  switch `emgain_off.request`.
- `newCallBack_emgain_off()` validates and dispatches an On request under the
  application's INDI mutex to the required derived `setEMGainOff()` method.
- The camera selects unity or disabled gain, updates `m_emGainSet`, and uses its
  normal gain setter. Cameras without EM gain need no additional implementation.
- The shared callback accepts this safe request regardless of temperature lock,
  publishes On/BUSY while invoking the camera, then resets Off with IDLE on
  success or ALERT on failure. It publishes the camera's selected numeric target
  without inventing a current gain measurement. Missing/Off requests do nothing.
- Camera availability and application of the setting remain camera-specific,
  consistent with existing gain setters. This is an explicit request, not an
  automatic response to temperature changes.

## Camera implementations and GUI

- OCAM, PICam, and cameraSim select logical gain 1 and reuse `setEMGain()`.
- Andor also selects logical gain 1; its existing setter sends SDK gain 0.
- PICam and Andor retain their conventional-amplifier behavior: no EM-gain SDK
  call is necessary while the selected amplifier is already non-multiplying.
- The shared camera GUI discovers `emgain_off.request` and displays an
  **EM gain off** button below the gain entry. It sends only the standard request,
  with no knowledge of the camera's numeric choice. Deletion/disconnection clears
  support; reconnect requires fresh capability and FSM information.

## Affected files

- `libMagAOX/app/dev/stdCamera.hpp` and `tests/stdCamera_test.cpp` beneath that directory.
- `apps/{ocam2KCtrl,picamCtrl,andorCtrl,cameraSim}/<app>.hpp`.
- New `apps/ocam2KCtrl/tests/ocam2KCtrl_emgain_test.cpp` and
  `apps/cameraSim/tests/cameraSim_emgain_test.cpp`; extended
  `apps/picamCtrl/tests/picamCtrl_test.cpp`.
- `gui/widgets/camera/camera.hpp`, plus new `tests/camera_test.cpp` and
  `tests/camera_test.pro` beneath that directory.
- `tests/tests.list`, `tests/Makefile.one`, and `tests/groups.dox`.
- This engineering record.

## Offline validation

All work runs on workstation `mx11`. No instrument commands, service restarts,
or hardware tests were performed.

| Suite | Assertions | Cases | Result |
| --- | ---: | ---: | --- |
| stdCamera | 4219 | 15 | Passed |
| stdCamera ROI (no EM-gain interface) | 529 | 3 | Passed |
| camera GUI | 65 | 4 | Passed |
| OCAM EM-gain-off | 34 | 1 | Passed |
| OCAM baseline | 474 | 22 | Passed |
| PICam | 8822 | 7 | Passed |
| cameraSim EM-gain-off | 20 | 1 | Passed |
| cameraSim baseline | 107 | 4 | Passed |

The shared suite checks capability gating, invalid/Off/missing requests,
camera-selected 0 or 1, temperature/state bypass, failure propagation, retries,
and actual FIFO acknowledgments for repeated requests. The socket-free GUI
suite checks discovery, the exact emitted request, deletion/reconnect/readiness,
and non-overlapping layout with gain, synchronization, and averaging controls.
An offscreen screenshot was inspected.

Camera-specific tests exercise the real shared callback through OCAM's serial
`gain 1` command, PICam SDK gain 1 and conventional-amplifier handling, and the
simulator's current/target gain. OCAM serial and PICam SDK failures propagate to
the request state. Total: 14270 assertions across 57 cases. Andor's SDK is absent
here, so its unity-to-SDK-zero mapping is source-reviewed rather than SDK-built
or exercised.

Touched C++ files received the required full-file documentation pass and
clang-format; mutex lifetime comments retain `//mutex scope`. Andor's older
formatting accounts for most of its diff. Functional changes, documentation
cleanup, and formatting are separated in the commit history. A lexical check
found no non-comment token changes from formatting. The documentation pass only
added parameter names to previously unnamed telemetry declarations in Andor and
cameraSim.

## Boundaries

This is a momentary command, not a persistent gain inhibit across later mode or
gain changes. Existing gain-setter behavior is preserved, including OCAM's
transport-success acknowledgment without parsing the gain-command response and
the conventional-amplifier behavior noted above. The shared callback publishes
the target without claiming a measured current gain. Independent review found no
scoped functional blockers.

The issue #105 changes are recorded on `jrmales/camera-emgain-temperature`.
Earlier work was committed before beginning this feature, as listed above.
No GitHub comments, pull requests, or remote updates were made. Instrument
validation is left to the user.

## Suggested PR text

Title:

```text
Add a standard camera EM-gain-off request and GUI button
```

Description:

```md
This work was performed by GPT-6 (Codex) in response to the prompt: "adding a standard request switch for any camera that exposes EM gain that causes the EMG to be set to 1 or off (it's a camera specific callback so the camera decides what it means)".

Expose emgain_off.request for EM-gain cameras and dispatch it to a camera-specific setter, allowing unity/off requests even without temperature lock. Add the corresponding button to the shared camera GUI.

Addresses #105. Implementation details and offline validation are recorded in agents/plans/2026-10/camera-emgain-off.md.
```

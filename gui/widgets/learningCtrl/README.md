# Shared NN / PO4AO control panel

`learningCtrlGUI <companion-device>` is the native Qt client for the [approved requirements](../../../agents/plans/2026-10/nn-po4ao-gui-requirements.md). It reuses `xWidget`, `statusLabel`, the MagAO-X resource stylesheet and `multiIndiManager`/`multiIndiPublisher`. The default width is 500 pixels; layouts handle narrower windows. No web framework, shell commands, loop control or plotting panel is added.

This is the frontend implementation. The managed `po4aoTrainer` supervisor and runtime completion readbacks are still being implemented; the GUI remains disabled when its companion is absent. Building it does not make the instrument controls operational.

## Behavior

- The NN catalog comes from the NN device's `model` definition. Labels are displayed; element identifiers are sent only when Enable is pressed. A selected value is not claimed to be active. Requested/loaded/active NN filenames are available on the reconstructor status tooltip. The backend must perform the agreed disable/load/enable sequence.
- Switch reconstructor sends a destination and expected source from observed state; it ignores the staged engine. Unknown/conflict/transition state disables that request. Learning, recollection and loop state never gate reconstruction controls.
- Link modulation sends a preference to the companion. The checkbox reflects readback; it issues no modulation motion itself.
- Enable/Disable PO4AO and Start/Pause/Resume learning labels follow the companion's effective state. A retained policy ID alone never implies PO4AO output is active. Disable's pause behavior, Reset's termination, and recollection noise/resume sequencing belong to the companion/runtime.
- Local request latches prevent repeated clicks before acknowledgement. Busy is followed by Ok or Alert after the backend has updated its result state. A ten-second failure to receive initial Busy/Alert reports unknown state, keeps the action latched and never retries. Once Busy is acknowledged, the backend owns the operation deadline. Reset remains available during ordinary learning/collection operations.
- Disconnect and device/property deletion invalidate cached capabilities. Reconnection performs discovery, never command replay. Sparse updates preserve catalog definitions and the user's unsent selection. Changing the NN mapping drops the old catalog. Closing the GUI does not control training or inference.

## Concrete companion contract

All commands target the named companion. INDI types/permissions and required elements are checked from definitions. The GUI subscribes to `fsm`, `devices`, the command properties below, `status`, `progress`, and `training_config`. It discovers the NN device from `devices.nn` and subscribes to that device's `fsm`, `model`, and `model_status` through the existing publisher.

| Property | Type / elements | Request semantics |
| --- | --- | --- |
| `nn_activate` | Read/write Text: `model` | Catalog element ID, never a constructed engine pathname. |
| `reconstructor` | Read/write Text: `target`, `expected` | Both `nn` or `cacao`; reject a stale expected source. This makes the plan's proposed selection request explicit. |
| `link_modulation` | Read/write Switch: `toggle` | Shared preference; no immediate modulation motion. |
| `po4ao_enable` | Read/write Switch: `toggle` | Explicit enabled/disabled target, with coordinated pause on disable. |
| `reset`, `recollect` | Read/write Switch: `request` | On requests the corresponding operation. |
| `learning` | Read/write Switch: `run`, `pause` | Explicit run/pause selection; Run starts or resumes according to the managed session. |

Every accepted request must publish Busy before work and then Ok/Alert after actual completion/failure. Set effective status before the terminal command acknowledgement. Return Alert for a rejected conflicting/stale request. A request vector is not an effective-state readback. Multiple clients must be serialized/idempotent in the companion; local UI latching is not backend serialization.

`fsm.state` is `READY` or `OPERATING` when the companion is usable. `status` is read-only Text with:

| Element | Values / meaning |
| --- | --- |
| `reconstructor` | `nn`, `cacao`, `unknown`, `conflict`, `transition`; derive from actual writer evidence. |
| `reconstruction_phase` | `idle`/`error` permit a new request; other values describe an in-progress handover/load. |
| `controller` | `integrator` when PO4AO is disabled; `po4ao` for effective learned control; `fallback` for enabled PO4AO awaiting a usable policy; `unknown` otherwise. |
| `learning` | `idle`, `starting`, `running`, `waiting`, `pausing`, `paused`, `completed`, `resetting`, `error`. |
| `training_phase` | `idle`/`error` permit ordinary requests; transitions include `enabling`, `disabling`, `resetting`, `recollecting`. Keep this independent of reconstruction. |
| `collection` | `ready`, `collecting`, `interrupted`, `unknown`. |
| `active_policy`, `retained_policy` | Effective and eligible retained model identities, respectively. |
| `modulation` | Readable observed modulation state, or Unknown. |
| `message`, `error` | Current operation/result and persistent error details. |

`progress.iteration` is a read-only number. `training_config.error` carries validation errors; `training_config.pending_reset` is empty or the pending revision requiring Reset. Other desired/applied revisions and collection progress may be added by the backend without changing the request semantics. Missing readbacks are not filled with presumed success. A connected INDI transport is distinguished from a ready companion; unchanged property values are not treated as disconnection.

This contract does not implement notebook ownership, leases or locks. It does not change the operator's responsibility for loop state or notebook/app use.

## Build and verification

Use the existing GUI app build/install flow in `gui/apps/learningCtrlGUI`. The only aggregate-build change is adding `learningCtrlGUI` to `all_guis`; role-based exclusions and the startup window script stay unchanged. Installation and running against the instrument remain operator actions.

Offline tests reuse the upstream socket-free publisher:

```sh
cd gui/widgets/learningCtrl/tests
qmake learningCtrl_test.pro
make -f makefile.learningCtrl_test
QT_QPA_PLATFORM=offscreen ./learningCtrl_test
```

Build the existing INDI static libraries first, as for other GUI tests. The optional `[.preview]` case saves a screenshot to `LEARNINGCTRL_SCREENSHOT` using deterministic fictional readbacks. It never creates an INDI socket. Test coverage includes all buttons, staged selection, Busy/Ok/Alert handling, missing acknowledgement, independent control domains, two subscribers on one publisher, deletion/redefinition, NN remapping, permissions, disconnect, config errors and 500/420-pixel layouts.

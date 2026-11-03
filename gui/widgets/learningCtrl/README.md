# Shared NN / PO4AO control panel

`po4aoGUI [po4ao-device [trainer-device [nn-device]]]` is the native Qt client for the [approved requirements](../../../agents/plans/2026-10/nn-po4ao-gui-requirements.md). It reuses `xWidget`, `statusLabel`, the MagAO-X resource stylesheet and `multiIndiManager`/`multiIndiPublisher`. The default width is 500 pixels; layouts handle narrower windows. No web framework, shell commands, loop control or plotting panel is added.

Defaults are `po4ao`, `po4aoTrainer`, and `nnReconstructor`. This single launcher replaces `learningCtrlGUI`.
The C++ runtime is subscribed independently of the optional trainer. Without a ready
trainer, native inference controls remain available and Start/Pause is unavailable.
NN controls independently subscribe to the configured NN device and remain available
without either PO4AO app. The panel has nnReconstructor and PO4AO section titles. An existing live training notebook needs no package migration.
With a live trainer mapped to this runtime, the same PO4AO controls use its coordinated
requests and learning becomes available in the same window. Arrival/loss sends no commands. See [deployment instructions](../../../apps/po4aoTrainer/DEPLOYMENT.md) for installation, configuration and the required operator-run target acceptance.

## Behavior

Without the trainer, Enable/Disable writes `control_enable.toggle`; Reset always
sends Off followed by `reset_model.toggle=On`; Recollect sends
`recollect_warmup.toggle=On`. Enable selects the resident engine; notebook export/reload
remains notebook code. Readbacks use native `fsm`, `model_status`, `collection_status`
and switch acknowledgements. No button controls notebook execution or Python objects.
Loaded and effective model IDs are shown separately. Native Disable remains available
during recollection. Native Reset supersedes pending selection without waiting for
frames. Its completion requires a newer applied reset generation, matching control
request/applied generations and observed control Off/Ok. The idempotent Off request
needs no separate Busy edge.

NN commands always target `nnReconstructor`; managed PO4AO commands target the optional trainer:

- The NN catalog comes from the NN device's `model` definition. Labels are displayed; element identifiers are sent only when Set reconstructor is pressed. A selected value is not claimed to be active. Requested/loaded/active NN filenames are available on the reconstructor status tooltip. The backend performs the agreed disable/load/enable sequence; selecting an already loaded disabled model only requires re-enabling it. Selecting the already active model is a no-op. NN-to-NN replacement leaves CACAO/modulation unchanged.
- Switch to CACAO / Switch to CNN sends a destination and expected source from observed state; it ignores the staged engine. Unknown/conflict/transition state disables that request. Learning, recollection and loop state never gate reconstruction controls.
- Confirmed stopped writers, including a quiescent NN fault, expose Switch to CACAO and selected-model Set reconstructor as recovery actions. A fault alone is not successful activation or proof that output has stopped.
- Link modulation sends a preference to the NN app. The checkbox reflects readback; it issues no modulation motion itself.
- Enable/Disable PO4AO and Start/Pause/Resume learning labels follow the companion's effective state. A retained policy ID alone never implies PO4AO output is active. Disable's pause behavior, Reset's termination, and recollection noise/resume sequencing belong to the companion/runtime.
- Disable remains available during publication/recollection when PO4AO is observed enabled, using the backend's superseding pause path. Enable stays serialized. A persistent pending Reset blocks ordinary training actions while leaving Reset available to retry. Collection displays frames/target beside its state.
- Device/property-scoped request latches prevent repeated clicks before acknowledgement.
  Busy is followed by Ok or Alert after the backend has updated its result state.
  A ten-second failure to receive initial Busy/Alert reports unknown state, keeps
  the action latched and never retries. Once Busy is acknowledged, the backend owns
  the operation deadline. When trainer availability changes, unresolved local
  requests, other clients' observed Busy operations and active managed phases block
  commands through the new route until completion or device/property invalidation.
  No cross-route supersession is attempted. Returning to the original route restores
  its normal coordination. Within the same route, managed Reset/Disable retain
  their existing superseding behavior; native Reset supersedes selection, and
  native Disable remains available during recollection. Reset remains available
  during ordinary learning/collection operations.
- Disconnect and device/property deletion invalidate cached capabilities. Reconnection performs discovery, never command replay. Sparse updates preserve catalog definitions and the user's unsent selection. Trainer mapping changes never alter NN subscriptions. Closing the GUI does not control training or inference.
- An advancing `status.heartbeat` is required after discovery. Five seconds without advancement marks the companion stale and disables its managed controls. Native PO4AO controls independently require an advancing `model_status.sequence` and defined native capabilities; a changed `model_status.instance` invalidates previous-process state and requests fresh definitions. Repeated cached heartbeat values do not renew freshness; unchanged controller/model values are otherwise normal.

## Device contracts

Managed commands target the named companion; the direct runtime fallback above is separate. INDI types/permissions and required elements are checked from definitions. The GUI subscribes to `fsm`, `devices`, the command properties below, `status`, `progress`, and `training_config`. The independently configured NN device supplies `fsm`, `model`, `model_status`, its three command vectors and `reconstruction_status`. The latter contains `reconstructor`, `phase`, `error`, `modulation`, `heartbeat`, and `instance`. An advancing heartbeat and the current instance are required independently of trainer availability.

| Property | Type / elements | Request semantics |
| --- | --- | --- |
| NN `nn_activate` | Read/write Text: `model` | Catalog element ID, never a constructed engine pathname. |
| NN `reconstructor` | Read/write Text: `target`, `expected` | Target `nn` or `cacao`; expected source `nn`, `cacao` or verified `stopped`. Reject a stale expected source. |
| NN `link_modulation` | Read/write Switch: `toggle` | Shared preference; no immediate modulation motion. |
| `po4ao_enable` | Read/write Switch: `toggle` | Explicit enabled/disabled target, with coordinated pause on disable. |
| `reset`, `recollect` | Read/write Switch: `request` | On requests the corresponding operation. |
| `learning` | Read/write Switch: `run`, `pause` | Explicit run/pause selection; Run starts or resumes according to the managed session. |

Every accepted request must publish Busy before work and then Ok/Alert after actual completion/failure. Set effective status before the terminal command acknowledgement. Return Alert for a rejected conflicting/stale request. A request vector is not an effective-state readback. Multiple clients must be serialized/idempotent in the owning backend; local UI latching is not backend serialization.

`fsm.state` is `READY` or `OPERATING` when the companion is usable. `status` is read-only Text with:

| Element | Values / meaning |
| --- | --- |
| `heartbeat` | Advancing companion loop sequence, independent of changes in other status fields. |
| `controller` | `integrator` when PO4AO is disabled; `po4ao` for effective learned control; `fallback` for enabled PO4AO awaiting a usable policy; `unknown` otherwise. |
| `learning` | `idle`, `starting`, `running`, `waiting`, `pausing`, `paused`, `completed`, `resetting`, `error`. |
| `training_phase` | `idle`/`error` permit ordinary requests; transitions include `enabling`, `disabling`, `resetting`, `reset_pending`, `recollecting`, `publishing`. Keep this independent of reconstruction. |
| `collection` | `ready`, `collecting`, `interrupted`, `unknown`. |
| `collection_frames`, `collection_target` | Current and required frames for the observed collection. |
| `active_policy`, `retained_policy` | Effective and eligible retained model identities, respectively. |
| `message`, `error` | Current operation/result and persistent error details. |

`progress.iteration` is a read-only number. `training_config.error` carries validation errors; `training_config.pending_reset` is empty or the pending configuration revision requiring Reset, distinct from an unfinished runtime Reset. Missing readbacks are not filled with presumed success. A connected INDI transport is distinguished from a ready companion through the heartbeat; unchanged controller/model values are not treated as disconnection.

Neither interface implements notebook ownership, leases or locks. It does not change the operator's responsibility for loop state or notebook/app use.

## Build and verification

Use the existing GUI app build/install flow in `gui/apps/po4aoGUI`. The only aggregate-build change is adding `po4aoGUI` to `all_guis`; role-based exclusions and the startup window script stay unchanged. Installation and running against the instrument remain operator actions.

Offline tests reuse the upstream socket-free publisher:

```sh
cd gui/widgets/learningCtrl/tests
qmake learningCtrl_test.pro
make -f makefile.learningCtrl_test
QT_QPA_PLATFORM=offscreen ./learningCtrl_test
```

Build the existing INDI static libraries first, as for other GUI tests. The optional `[.preview]` case saves a screenshot to `LEARNINGCTRL_SCREENSHOT` using deterministic fictional readbacks. It never creates an INDI socket. Test coverage includes all buttons, staged selection, Busy/Ok/Alert handling, missing acknowledgement, independent control domains, two subscribers on one publisher, deletion/redefinition, independent NN discovery, permissions, disconnect, config errors and 500/420-pixel layouts.

Review regressions additionally cover stopped/fault recovery, Disable while publishing
or recollecting, collection frame progress, heartbeat expiry and fresh recovery.

Native-only regressions cover no trainer definitions, enable/disable/recollection,
Reset with control On/Off, late trainer arrival/departure, independent heartbeat
expiry, process restart, native permissions and unresolved cross-device requests.

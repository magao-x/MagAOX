Prompt: We need to implement a version of apps/zaberLowLevel that uses the binary protocol of zaber instead of ASCII. The new app `zaberLowLevelBinary` should function identically to the ASCII version as much as possible. A previous implementation of a similar app can be found in `../VisAO/zaberStage`. I can also provide the protocol reference documents if needed.

## Analysis

The existing `apps/zaberLowLevel` app already contains almost all of the MagAO-X-facing behavior we want to preserve:

- configuration loading through `tty::usbDevice`
- stage discovery from config sections keyed by serial number
- INDI property schema used by `zaberCtrl`
- stage state persistence in `m_sysPath/<configName>/<stageName>`
- FSM / power-management behavior inherited from `MagAOXApp`

The protocol-specific logic is currently spread across two places:

- `apps/zaberLowLevel/zaberLowLevel.hpp`
  - connection setup
  - chain renumbering
  - stage enumeration via `get system.serial`
- `apps/zaberLowLevel/zaberStage.hpp`
  - command formatting/parsing
  - status, warnings, homing, parking, temperature, max position, motion commands

There is already a bundled Zaber binary C transport implementation in this tree:

- `apps/zaberLowLevel/zaber-core-serial-c-v1.0/zb_serial.[ch]`

The older `../VisAO/zaberStage` code is useful mainly as a reference for:

- binary framing and `zb_*` API usage
- command numbers for home / move / stop / status / position / device mode / max position
- expected binary reply flow and timeout handling

It is not a drop-in reuse candidate because it is single-device oriented and does not implement the current MagAO-X INDI interface, config shape, warning bookkeeping, or multi-stage chain mapping by serial number.

We now also have the correct protocol/manual context for the target hardware:

- hardware: `T-LSM050B-S-KT03`
- confirmed firmware: `5.35`
- authoritative command reference: `/home/jrmales/Documents/MyPapers/Projects/MagAOX/Electronics/Zaber/zaber/T-LSM_Users_Manual-Zaber.pdf`

This matters because the generic Zaber Binary Protocol Manual we first reviewed is written around firmware `6.xx` and points firmware `5.xx` users back to the product manual. The `T-LSM` manual explicitly covers firmware `5.00 and up` and is therefore the correct implementation reference for this app.

## Implementation Plan

### 1. Create a new app by cloning the current low-level interface shape

Add a sibling app `apps/zaberLowLevelBinary` instead of mutating `apps/zaberLowLevel` in place.

Reasoning:

- `zaberCtrl` already supports selecting a low-level app name through `stage.lowLevelName`
- this lets us preserve the current ASCII implementation during bring-up
- it minimizes operational risk and makes side-by-side comparison possible

Initial file set:

- `apps/zaberLowLevelBinary/Makefile`
- `apps/zaberLowLevelBinary/zaberLowLevelBinary.cpp`
- `apps/zaberLowLevelBinary/zaberLowLevelBinary.hpp`
- `apps/zaberLowLevelBinary/zaberBinaryStage.hpp`
- `apps/zaberLowLevelBinary/tests/...`

Also vendor or reference the binary transport sources in the new app build:

- prefer compiling `zb_serial.c`
- reuse `z_common.h` / `zb_serial.h`

### 2. Preserve the MagAO-X and INDI contract exactly where possible

The new app should expose the same externally visible properties and semantics as `zaberLowLevel`:

- same power-management behavior
- same config discovery pattern for stage sections by serial number
- same INDI properties:
  - `curr_state`
  - `max_pos`
  - `parked`
  - `last_homed`
  - `curr_pos`
  - `temp`
  - `warning`
  - `tgt_pos`
  - `req_home`
  - `home_all`
  - `req_halt`
  - `req_ehalt`
- same state-file format if feasible
- same stage-name keyed INDI elements

This is important so `apps/zaberCtrl` can target `zaberLowLevelBinary` by configuration only, with no code changes required in the controller.

### 3. Split protocol-independent stage state from binary transport details

Do not copy the current ASCII `zaberStage` blindly. Instead, use the current `zaberStage` responsibilities as the interface target and re-implement the protocol layer for binary.

Suggested structure:

- `zaberLowLevelBinary`
  - owns the serial port
  - performs connection / reset / bus discovery
  - maintains maps:
    - serial -> configured stage index
    - address -> active stage index
    - name -> stage index
- `zaberBinaryStage`
  - stores MagAO-X-visible state:
    - name, serial, address, axis
    - position, target, max position
    - homing, parked, warnings, temperature, last homed
  - exposes stage operations analogous to the ASCII version:
    - `getMaxPos`
    - `getParked`
    - `updatePos`
    - `updateTemp`
    - `enableKnob`
    - `enableLED`
    - `stop`
    - `estop`
    - `home`
    - `park`
    - `unpark`
    - `moveAbs`
    - warning/state helpers

This keeps most of the current app logic readable and makes it easier to compare method-for-method against the ASCII implementation.

### 4. Implement binary connection and discovery carefully

The biggest design question is how to reproduce the ASCII startup sequence:

- ASCII app:
  - connect
  - drain
  - renumber
  - query `system.serial`
  - match discovered serial numbers to configured stage sections

Binary app needs an equivalent discovery path. The likely approach, based on the bundled binary API and old VisAO code, is:

- connect with `zb_connect`
- optionally set timeout explicitly with `zb_set_timeout`
- issue global/broadcast renumber if supported by the binary protocol
- probe device addresses after renumber to read serial number and identify configured devices

The plan here is to implement discovery in two phases:

1. Minimal viable discovery:
   - renumber chain
   - scan a bounded address range
   - query each responding device for serial number
   - map serial number to configured stage

2. Hardened discovery:
   - stop scan once a reasonable consecutive-miss threshold is reached
   - log unknown but present devices
   - clearly distinguish configured-but-missing vs discovered-but-unconfigured

For the target `T-LSM` firmware `5.35`, the product manual confirms that `Return Serial Number` is available as `Cmd 63`, so serial-number based discovery is a valid plan.

### 5. Port core device commands from ASCII text to binary command IDs

From the old `VisAO/zaberStage` implementation, the following binary commands appear directly relevant:

- `1`: home
- `20`: move absolute
- `21`: move relative
- `23`: stop
- `53`: return setting
- `54`: return status
- `60`: return current position

The old code also uses setting numbers:

- `40`: device mode / flags, including homed bit and reply/LED-related bits
- `44`: max position

The implementation should define named constants or a small enum for these command IDs and setting numbers instead of scattering raw integers.

For the target `T-LSM` firmware `5.35`, the local manual confirms these additional points:

- all commands are 6-byte binary packets at `9600 8N1`
- `Cmd 2` renumber remains valid, but for firmware `5xx` device numbers persist across power cycles
- `Cmd 51` returns firmware version
- `Cmd 63` returns serial number
- `Cmd 40` `Set Device Mode` includes the manual-knob control bits

### 6. Reconcile feature gaps between ASCII and binary

The current ASCII app does more than the old VisAO binary controller:

- reads warning state and logs per-warning flags
- tracks parked / unparked state
- reads temperature
- disables the knob every startup
- persists last known stage state to disk

Before coding the full FSM, verify which of these are available in the binary protocol for the target hardware:

- serial number query
- warning retrieval
- parked state query and park/unpark commands
- temperature query
- knob-disable / lockout support

Updated target-hardware conclusions from the `T-LSM` manual and your notes:

- serial number query: available via `Cmd 63`
- knob disable: available via `Set Device Mode (Cmd 40)`, specifically using `bit_3`
- parked state: not native in the same way as the ASCII app; we should emulate parking using stored positions
- temperature: still needs verification for this exact device/firmware path
- warning retrieval: still needs verification against firmware `5.35` behavior and current operational needs

Implementation policy:

- if a feature exists in binary, implement it and preserve behavior
- if a feature does not exist, keep the property but degrade gracefully
  - example: fixed default for `parked` or `temp`
  - log once per stage that the value is unavailable in binary mode
- do not silently fake safety-critical behavior

Parking plan for this app:

- preserve the `parked` INDI/state-file concept for compatibility
- implement "park" as a move to a configured or persisted parked raw position
- implement "unpark" as clearing the parked bookkeeping state after a successful move away from the parked position
- document clearly that this is MagAO-X parking semantics layered on top of Zaber motion control, not a native device park latch unless the firmware proves otherwise

### 7. Reuse as much of the existing FSM as practical

The current `appLogic()` sequencing is already appropriate for MagAO-X operations:

- `POWERON`
- `NODEVICE`
- `NOTCONNECTED`
- `CONNECTED`
- `READY`
- power-off handling

Plan:

- copy the ASCII FSM into the new app as the starting point
- replace only the protocol-dependent calls
- preserve locking and INDI update sequencing
- preserve state-file read/write behavior

This should keep operational behavior aligned and reduce surprises for downstream clients.

### 8. Keep the initial implementation single-axis per device unless proven otherwise

The current app defaults `m_axisNumber` to `0` and operationally treats each configured stage as one logical axis. The old VisAO code also targets a single axis per device.

Plan:

- keep the same assumption for v1 of `zaberLowLevelBinary`
- make axis number explicit in the stage class so future extension is easy
- do not expand to general multi-axis support unless the target hardware requires it

### 9. Add focused tests around behavior, not just protocol calls

The current `zaberLowLevel` tests are light and partly stale, so the binary app should at least add unit coverage where protocol mapping is easy to regress.

Priority test areas:

- config section parsing into named stages
- mapping discovered serial numbers to configured stages
- state-file read/write compatibility
- INDI property creation and callback plumbing
- command encoding wrappers around binary operations where mockable
- stage status interpretation:
  - moving
  - homing
  - homed
  - unknown / not connected

If direct mocking of `zb_*` is awkward, factor binary I/O behind small helper methods so tests can override them in a harness subclass.

### 10. Stage rollout in two implementation passes

Pass 1:

- compile new app
- connect to device
- discover configured stages by serial
- read position / max position
- support move, home, stop
- disable the manual knob using `Set Device Mode` bit 3
- expose compatible INDI properties

Pass 2:

- warnings
- parked state
- temperature
- emergency stop semantics
- operational polish and fuller tests

This should get us to hardware validation faster while keeping the plan realistic.

## File-Level Work Plan

### `apps/zaberLowLevelBinary/Makefile`

- set `TARGET=zaberLowLevelBinary`
- compile/link `zb_serial.o`
- mirror the current app build structure

### `apps/zaberLowLevelBinary/zaberLowLevelBinary.hpp`

- copy the current class skeleton from `zaberLowLevel`
- rename class / include guard / Doxygen groups
- swap ASCII transport include for binary transport include
- keep public INDI surface and FSM method structure aligned

### `apps/zaberLowLevelBinary/zaberBinaryStage.hpp`

- implement the stage state container and binary command wrappers
- use named constants for command IDs/settings
- preserve documentation and declaration style from `agent_context.md`

### `apps/zaberLowLevelBinary/zaberLowLevelBinary.cpp`

- minimal main program only, matching existing app conventions

### `apps/zaberLowLevelBinary/tests/*`

- start by cloning the current low-level test scaffolding
- update for new class names
- add test seams around binary transport behavior

## Risks and Open Questions

### Highest-risk items

- availability of warning, parked, and temperature features in binary mode
- whether all target Zaber devices support the same binary command set as the old VisAO hardware
- defining robust MagAO-X "parking" semantics on top of firmware `5.35` motion commands without surprising operators

### Questions to resolve during implementation

- Is the binary app expected to read the exact same config file format as `zaberLowLevel`?
  - ANSWER: if new/different config options are needed it's ok
  - Plan update: preserve the current config format by default, but allow new binary-only options if they materially improve reliability. Likely candidates are:
    - parked raw position
    - renumber-on-connect enable/disable
    - discovery max address / timeout tuning
- Should `zaberCtrl` remain unchanged and switch by `stage.lowLevelName` only?
  - ANSWER: yes
- Are there deployed stages where ASCII-only features were relied on operationally?
  - ANSWER: 
      - we will need to hack the "parking" using stored positions
      - knob can be disabled with "Set Device Mode" bit_3

### Resolved assumptions

- `zaberCtrl` should remain unchanged and select the backend through `stage.lowLevelName`
- serial-number discovery is supported on the target firmware/hardware via `Cmd 63`
- parking should be implemented in MagAO-X as a compatibility layer using stored positions
- knob disable should be implemented early using `Set Device Mode (Cmd 40)` bit 3
      - I don't see an obvious parallel to warnings
      - looks like temperature is not available

### When to request protocol docs

I do not need the docs to start scaffolding the new app and porting the FSM/interface. I would want the Zaber binary protocol reference before locking in:

- device discovery by serial number
- warning retrieval mapping
- parked / unparked support
- temperature and knob-disable equivalents
- emergency stop semantics

## Recommended Execution Order

1. Scaffold `apps/zaberLowLevelBinary` by copying the current MagAO-X app shape.
2. Replace ASCII transport with `zb_*` transport and get a successful connect/disconnect build.
3. Implement binary discovery and serial-number mapping using `Cmd 63`, with firmware `5.35` timing/renumber behavior from the `T-LSM` manual.
4. Port motion/status primitives into `zaberBinaryStage`.
5. Implement manual-knob disable via `Set Device Mode (Cmd 40)` bit 3 during startup.
6. Reconnect the existing FSM and INDI publication flow.
7. Implement MagAO-X parking semantics via stored positions and explicit bookkeeping.
8. Add tests for discovery, stage-state interpretation, parking bookkeeping, and callback wiring.
9. Validate remaining parity features, then either implement or document graceful degradation.

## Deliverable Definition

The plan is complete when `zaberLowLevelBinary`:

- builds as a separate MagAO-X app
- can be selected by `zaberCtrl` through configuration only
- discovers configured stages by serial number
- publishes the same core INDI interface as `zaberLowLevel`
- supports home / move / halt with equivalent operational behavior
- clearly documents any remaining binary-vs-ASCII feature gaps

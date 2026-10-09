# Task Description

Kinetix Hotplug

## Problem Statement
The Teledyne Kinetix cameras (camllowfs and camflowfs) require PCIe hotplug operations after power on.

## Discussion
The Kinetix cameras are read by PCIe expansion cards, and as such are part of the PCIe bus.  This means they need to be on when the computer is booted to be recognized.  After that boot, the cameras can be power cycled but if so they need to be "hotplugged" to be reinitialized.  The vendor provides this script: /opt/pvcam/drivers/in-kernel/pcie/hotplug_pcie.sh.

This is not part of normal MagAO-X power control operations.  See apps/pvcamCtrl for the relevant controller.  To be able to power down the cameras when not in use, we want to have this hotplugging occur as part of the power on logic in pvcamCtrl.

## Requirements
1. pvcamCtrl should handle execution of the needed commands to reinitialized the PCIe device

2. it must do this for one camera without interfering with the other camera. 

3. pvcamCtrl must gracefully handle the case where the camera is not found, which will occur on a reboot if the camera was not powered

4. pvcamCtrl must be at 100% line coverage.  This does not include libMagAOX includes, this is handled elsewhere.

## Tests and Metrics
Add tests following usual MagAO-X practices for apps.  Recently standardized apps include zaberLowLevel and virtualPDU.

Key logic to test is power-off/power-on with and without cameras.  

## Scope and Caveats
Changes to libMagAOX, e.g. stdCamera are in scope.  Test coverage of anything outside apps/pvcamCtrl is out of scope.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan
<!-- This section will be filled out by the agent -->

Status (2026-10-09): planning only. Claude Opus 5.5 prepared this plan. Implementation waits for user review and answers to the questions below.

Update (2026-10-09, after the user answered Q1–Q9):
- **Decisions:**
  - explicit BDF;
  - hotplug at POWERON after 15 s, with a 30 s NODEVICE retry;
  - no PCI work at power-off;
  - fix both power-handling bugs and the coverage blockers;
  - lock at `<m_sysPath>/pvcamCtrl_pcie.lock`.
- **Q5 result:** a running pvcamCtrl holds only its own `/dev/pvcamPCIE_N`. Removing our stale camera child is therefore safe while the other app operates, provided the shared lock keeps the other instance's `connect()` enumeration out.
- **PVCAM SDK headers** are now installed locally for stub verification.
- **Still open:** Q10, the instrument check of link state and config-space readings, plus Q11 on whether to start before Q10.

**Design addition:** stale-device detection reads the camera's PCI config space directly, instead of saving state to disk (the alternative suggested via flipperCtrl parking). See Requirements 2 and 4.

## Task Summary
<!-- The agent should summarize the task as they understand it -->

When its power channel turns on, `pvcamCtrl` should re-enumerate the PCIe camera for its own Kinetix and then connect to it as usual. It must not reset, remove, or otherwise disturb the other Kinetix. If the camera does not appear, the app stays in `NODEVICE`: no error spam, no shutdown, and it retries. The work also brings `apps/pvcamCtrl` to 100% line coverage with an offline Catch2 suite in the zaberLowLevel/virtualPDU style.

### Repository and vendor findings

| Area | Finding | Consequence |
| --- | --- | --- |
| Vendor script `/opt/pvcam/drivers/in-kernel/pcie/hotplug_pcie.sh` | It scans every supported card (OSS, IOI, Dolphin). It then performs a secondary bus reset (BRIDGE_CONTROL bit `0x40`, 10 ms, restore, 500 ms) on **every** downstream port, removes **every** camera (`1b6b:0001`) through sysfs `remove`, and `rescan`s every downstream port. It refuses to run if `pvcam_pcie` refcnt > holders, i.e. if **any** camera device is open. | Calling it from one app would reset the other camera's link, and it refuses while the other camera is open. It cannot satisfy Req. 2. The plan reimplements the same three steps scoped to **one** downstream port. |
| Script comment "Camera names cannot be mapped to ports" | `/dev/pvcamPCIE_N` assignment is probe-order dependent. | Per-camera targeting must use the PCI downstream port, not the PVCAM camera name. `connect()` already finds the camera by serial number, so renaming after a rescan is harmless. |
| Script warning "Removing camera while open causes now kernel oops" | Removal is safe only if no process has that camera's device node open. | Before removal: close our own handle, call `pl_pvcam_uninit()`, and check that no other process has the device open (see Q1/Q5). |
| Privileges | Apps install as setuid root (`Make/magAOXApp.mk`: `--mode=4755 --owner=root`). `MagAOXApp::elevatedPrivileges` (RAII) is already used by several apps. | sysfs `remove`/`rescan` and config-space writes happen inside an `elevatedPrivileges` scope. No sudoers entry and no external `setpci` are needed: config space is written via `pwrite` to `/sys/bus/pci/devices/<port>/config` at offset `0x3E`. |
| Power sequencing in `pvcamCtrl::appLogic()` | The POWERON branch waits `powerOnWaitElapsed()` (15 s) and then sets `NOTCONNECTED` **before** `STDCAMERA_APP_LOGIC`. stdCamera's POWERON block therefore never runs, and `powerOnDefaults()` and the ROI reset are skipped on power-on. `ocam2KCtrl` runs `STDCAMERA_APP_LOGIC` first and avoids this. | This is a pre-existing bug at the exact place where the hotplug hook goes (Q4). |
| `onPowerOff()`/`whilePowerOff()` | pvcamCtrl overrides neither. stdCamera, frameGrabber, and dssShutter all document that the derived app must call their versions. The PVCAM handle stays open across a power-off. | This is a pre-existing omission (Q4). It also determines when it is safe to close the handle and touch the PCI device. |
| Framegrabber thread | It leaves acquisition when `powerState() <= 0`, calls `reconfig()` (`pl_exp_stop_cont`), and parks while the state is not READY/OPERATING. Nothing signals that it is idle. | Closing the handle in `onPowerOff()` would race `reconfig()` on the fg thread. At POWERON plus 15 s the thread is parked, so doing all handle and PCI work then is race-free. |
| Concurrent `pvcamCtrl` instances | While in NODEVICE/NOTCONNECTED, `connect()` runs every loop. Each run does `pl_pvcam_init`, enumerates, and `pl_cam_open(OPEN_EXCLUSIVE)`s **every** camera to read serials. | The other instance can briefly open our camera at any time, including mid-removal. A cross-process `flock` should serialize `connect()` enumeration against hotplug. No `flock` precedent exists in the repo. |
| PVCAM SDK on workstation | `/opt/pvcam/sdk/include` and `/opt/pvcam/library` are absent locally. Only the driver script is present. | Tests need stand-in `master.h`/`pvcam.h` and fake `pl_*` functions, following the `tests/edtinc.h` precedent and the macro-redirect pattern in `zaberLowLevel_power_test.cpp`. Vendor headers will not be committed. |
| Instrument topology (Q1 output in `kinetix-hotplug-files/`, 2026-10-09) | One Dolphin PXH832 (`10b5:8733`) with upstream port `0000:41:00.0` and two downstream ports. `0000:42:08.0` leads to camera `0000:43:00.0`, and `0000:42:09.0` leads to camera `0000:44:00.0`. Each camera is alone on its own secondary bus (`[43]`, `[44]`). | A secondary bus reset plus `rescan` on one downstream port is physically isolated from the other camera. The single-card layout does not change the per-port design. The upstream port `0000:41:00.0` must never be reset, because that would take down both cameras. |
| Device-node ↔ PCI mapping | The camera PCI directories have no `pvcam_pcie` child. `/sys/class/pvcam_pcie/pvcamPCIE_N` links to `/sys/devices/virtual/...`, not to the PCI device. The user reports that `N` is not stable between boots. | sysfs cannot tell which `/dev/pvcamPCIE_N` is ours, so neither an exact "device open elsewhere" check nor a serial ↔ port check is possible this way. A wrong `downstreamPort` in config would hotplug the **other** camera. A runtime interlock is needed (see Requirement 2 and Q10). |
| Existing tests | pvcamCtrl has no tests and is not in `tests/tests.list`. | A new suite, a `tests/Makefile.one` case (stub include path), a `tests.list` entry, and a `groups.dox` entry are needed. |
| Other pre-existing defects seen (coverage blockers) | `fillSpeedTable()`: returns `false` (= 0 = success) on `pl_get_enum_param` failure, queries `maxg` with `ATTR_MIN`, and uses `new(std::nothrow)` with unreachable null checks (also in `dumpEnum`). `getTemp()`: logs and sets ERROR on a failed read but then uses uninitialized `isAvailable`/`stemp`/`ctemp`. `configureAcquisition()`: the circular buffer uses `new(std::nothrow)`, whose failure path cannot be hit in a test. | 100% coverage requires either minimal fixes or `LCOV_EXCL` markers (Q6). |

## Key Assumptions
<!-- The agent should list any assumptions they have made -->

- Each Kinetix sits behind its own switch **downstream port**, so a secondary bus reset of that one port affects only that camera. Confirmed by the Q1 output: on the Dolphin PXH832, `0000:42:08.0` and `0000:42:09.0` each lead to exactly one camera.
- The downstream port stays enumerated when the camera is unpowered, as the script's comment tree shows ("camera (turned off)" under a present downstream port). A reboot without camera power therefore leaves a port with no camera child.
- Per the user (Q8), if the host booted with a camera unpowered, hotplug is not expected to recover it. A host reboot with the camera powered is required. The app's job in that case is only graceful `NODEVICE` with a clear, once-logged diagnostic. Retries continue at `retryInterval`; an SBR plus `rescan` of our own empty port is harmless.
- The serial ↔ port mapping from the Q2 answer comes from the vendor script's "probably" guess, which is based on unstable `pvcamPCIE_N` order. It is a starting configuration, to be confirmed by the interlock and the first hardware validation:
  - camflowfs `A22J723004` on `0000:42:08.0` (camera `0000:43:00.0`);
  - camllowfs `A22J723005` on `0000:42:09.0` (camera `0000:44:00.0`).
- The 15 s `m_powerOnWait` already in pvcamCtrl is long enough for the Kinetix PCIe link to come up before the rescan (Q3).
- Hotplug is opt-in per instance: if no downstream port is configured, behavior is unchanged. This preserves non-PCIe PVCAM use (e.g. USB cameras).
- Coverage target: all executable lines in `apps/pvcamCtrl/*.hpp`. The two-line `main()` in `pvcamCtrl.cpp` is not compiled into tests, consistent with other apps. libMagAOX includes are excluded per the task.
- Instrument-side steps (topology inspection, `lsof`, hardware validation of power cycles) are run by the user. The agent will not run anything on exao1/2/3/5 or through an INDI tunnel.
- The untracked/deleted `flipper-parking.md` rename in the worktree belongs to separate work and will not be touched or committed here.

## Requirements
<!-- The agent should list the requirements to which they are planning -->

1. **Scoped hotplug.** On power-on, after `m_powerOnWait`, pvcamCtrl re-enumerates only its configured downstream port, mirroring the vendor steps and timing:
   - secondary bus reset (set `0x40`, 10 ms, restore, 500 ms);
   - `remove` the camera child if one is present;
   - 500 ms;
   - `rescan` the port;
   - 500 ms;
   - verify that a `1b6b:0001` child exists.
2. **No interference.** No sysfs or config write ever targets another port or device, or the upstream port. Before removal, the app closes its own handle and uninitializes PVCAM, and takes a cross-process lock that `connect()` also takes.
   - Because device nodes cannot be mapped to ports, a **link-state interlock** guards against a wrong `downstreamPort`: when our power goes off, the configured port's link must drop (Data Link Layer Link Active clear in the port's PCIe Link Status register, read via `config`). If it does not, the app logs a critical misconfiguration and disables hotplug until restart.
   - At hotplug time, a port whose link was never seen down since our last power-off is treated the same way.
   - An equivalent check that needs no link-status support: while our power is off, a hardware read of the camera child's vendor ID through sysfs `config` must return `0xffff`.
   - Which of the two signals to use, or both, is pending Q10.
3. **Graceful absence.** If the port has no camera after hotplug, or the port is missing, the app goes to `NODEVICE` and logs once (`stateLogged()` pattern). Hotplug is retried at a configurable interval while power is on. If power or the power target goes off during the sequence, it aborts silently, following the existing `powerState() != 1 || powerStateTarget() != 1` pattern.
4. **Recovery without power cycle.**
   - **Stale-device detection.** Before any `connect()` while power is on, check whether the camera child under our port is stale. A power cycle resets the device, so the Memory Space Enable bit in its Command register (config offset `0x04`) reads clear, and/or BAR0 no longer matches the address the kernel assigned (`resource`).
   - **Why.** This covers a power cycle the app never saw (camera cycled while the app was down, or by hand). It hotplugs **before** PVCAM opens a stale device, whose behavior is unknown and could include a hang. It needs no state saved to disk and works across app restarts and host reboots.
   - **Diagnostic.** No camera child at all while power is on means the camera was never enumerated since boot, or a hotplug failed. That gets a once-logged diagnostic naming the reboot requirement.
   - **Fallback.** *Dropped during implementation (see Implementation Notes).* A healthy camera child whose serial is not found is never reset. That would require trusting the port mapping for a working device.
5. **Configuration.** New `[pcie]` keys, documented and validated at startup (port exists and its vendor:device is one of the script's four `CARD_IDS`):
   - `downstreamPort`: PCI address (BDF) of the switch downstream port, e.g. `0000:0a:00.0`; empty disables hotplug;
   - `retryInterval`: seconds.
6. **Coverage and style.** 100% line coverage of `apps/pvcamCtrl` headers, with tests documented per AGENTS.md §19–20. The app follows the header-only pattern, gets a full Doxygen pass on touched files, is clang-formatted, and is committed as functional, then docs, then format commits.

## Questions and Points of Clarification
<!-- The agent should list any open issues requiring user clarification -->

1. **Instrument PCI topology.** *Answered 2026-10-09; output is in `kinetix-hotplug-files/`.*
   - Both cameras are on one Dolphin PXH832: `0000:42:08.0` → `0000:43:00.0` and `0000:42:09.0` → `0000:44:00.0`.
   - `pvcamPCIE_N` class devices are virtual, with no link to the PCI device, and `N` is unstable across boots.
2. **Port selection by BDF.** Is an explicit `pcie.downstreamPort` BDF per instance acceptable? It is deterministic but would need editing if cards move slots. The alternative, auto-selecting "the port with no camera", is ambiguous when both cameras power on together. Recommendation: explicit BDF plus startup validation plus the link-state interlock (Requirement 2). Which camera, camllowfs or camflowfs, is on `0000:42:08.0` and which on `0000:42:09.0`?

Answer: yes BDF is acceptable.
The mapping here is "probably" according to hotplug_pcie.sh:
camflowfs A22J723004 is currently at pvcamPCIE_1 -> 0000:43:00.0 
camllowfs A22J723005 at currently at pvcamPCIE_0 -> 0000:44:00.0

3. **Trigger and timing.** Proposed: always hotplug on the POWERON transition after the existing 15 s wait. Also retry while in NODEVICE every `retryInterval`, recommended default 30 s; a hotplug of an empty port is non-disruptive. Is 15 s an adequate Kinetix boot time before rescan, and is 30 s a reasonable retry? Should the camera's PCI device be removed proactively at power-off instead? Recommendation: no. Do all PCI work at power-on, when the framegrabber thread is guaranteed parked, to avoid racing `reconfig()`.

Answer: yes to all.

4. **Pre-existing power-handling bugs.** May I fix these in scope? Both change existing behavior.
   - (a) Reorder `appLogic()` so stdCamera's POWERON block runs, which means `powerOnDefaults()` and the ROI reset actually happen on power-on.
   - (b) Add `onPowerOff()`/`whilePowerOff()` that call the stdCamera/frameGrabber/dssShutter versions (INDI blanking, `m_reconfig`) and mark hotplug pending.

   Recommendation: yes to both. They sit at the same hook point and the tests would otherwise encode the bugs.

Answer: yes to both.

5. **Other processes holding camera devices (user-run, still open).** With one camera app OPERATING, please run `sudo lsof /dev/pvcamPCIE_*`. If PVCAM holds file descriptors for **all** cameras, not just the opened one, then removing our stale camera while the other app runs could oops the kernel, and the cross-process lock is not enough. In that case the fallback is to skip `remove` and rely on SBR plus `rescan` only, which needs a hardware trial (Q8). Since the Q1 output shows nodes can't be mapped to ports, this result now decides the removal design.

Agent note: the result shows a running pvcamCtrl holds only its own camera's device node. `remove` of our stale camera stays in the design, protected by our own close/uninit plus the shared lock around the other instance's `connect()` enumeration.

Results:
```
[jrmales@exao3 ~]$ xctrl shutdown camllowfs
Waiting for tmux session for camllowfs to exit...
Waiting for tmux session for camllowfs to exit...
Waiting for tmux session for camllowfs to exit...
Ended tmux session for camllowfs
[jrmales@exao3 ~]$ xctrl status camllowfs
camllowfs: not started

[jrmales@exao3 ~]$ xctrl status camflowfs
camflowfs: running (pid: 341389)

[jrmales@exao3 ~]$ sudo lsof /dev/pvcamPCIE_* 
[sudo] password for jrmales: 
lsof: WARNING: can't stat() fuse.irodsfs file system /srv/cyverse
      Output information may be incomplete.
COMMAND      PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
pvcamCtrl 341389 xsup   12u   CHR  240,1      0t0 1353 /dev/pvcamPCIE_1
```

6. **Coverage blockers in existing code.** Fix minimally rather than add `LCOV_EXCL`?
   - `fillSpeedTable()`: return `-1` on error, query `ATTR_MAX` for `maxg`, use `std::vector<char>` instead of `new(nothrow)` (same in `dumpEnum`);
   - `getTemp()`: return after a failed read instead of using uninitialized values;
   - circular buffer: a `std::vector<uns8>` or an allocation seam.

   Recommendation: fix them. The changes are small, and `maxG` is only logged to stderr.

Answer: agree

7. **Lock file location.** Recommendation: `<m_sysPath>/pvcamCtrl_pcie.lock` (e.g. `/opt/MagAOX/sys/pvcamCtrl_pcie.lock`). It is shared by all pvcamCtrl instances on the host and taken non-blocking, with a retry on the next loop. Acceptable?

Answer: acceptable 

8. **Hardware validation (user-run, after implementation).** Proposed sequence on the instrument:
   - boot with both cameras off;
   - power on lowfs only, then flowfs while lowfs is OPERATING;
   - power-cycle lowfs while flowfs streams and confirm flowfs frames are uninterrupted;
   - restart the app with power already on.

   Any additional cases?

Answer: I think "boot with both cameras off" won't work.  If you mean "startup the app" instead of "boot the computer" then it does work.  

Remark: see the parking system in flipperCtrl, it's a conceptual idea for how this app can maintain any state it might want across app restarts.  We could have it check for reboots since last app shutdown, for instance.  Only if this helps.

Agent note:
- **Validation wording.** Agreed; I meant "start the app with both cameras powered off", not a host boot. The revised validation sequence is:
  - (a) start both apps with both cameras powered off (cameras enumerated at boot);
  - (b) power on lowfs only, then flowfs while lowfs is OPERATING;
  - (c) power-cycle lowfs while flowfs streams, and confirm flowfs frames are uninterrupted;
  - (d) power-cycle a camera while its app is shut down, then start the app with power on. This exercises stale detection;
  - (e) restart an app with its camera on and healthy. There should be no hotplug;
  - (f) optional, if convenient: reboot the host with one camera unpowered, then power it on. This confirms graceful NODEVICE and the diagnostic;
  - (g) confirm the port mapping: power off one camera, and its app must not report an interlock failure.
- **Persistence.** It would help with exactly one thing: knowing whether the device was power-cycled since the kernel enumerated it, for example across an app restart. Reading the camera's PCI config space answers that directly and more reliably, including for power changes the app never observed. So I do not plan a state file. If Q10 shows config-space reads are not informative, a flipperCtrl-style file in `<m_sysPath>/<configName>/` would be the fallback, holding the boot ID and a "power-off seen since last enumeration" flag.

9. **PVCAM SDK headers on workstation.** Could `/opt/pvcam/sdk/include` be installed locally (not committed)? Then the stub declarations' signatures could be checked against the real ones before the first instrument build. Otherwise the stubs are written from the documented API, and mismatches would only show up when the user builds on the instrument.

Answer: done

10. **Link-state visibility (user-run, read-only).** Please capture `sudo lspci -vv -s 0000:42:08.0` and `sudo lspci -vv -s 0000:42:09.0`, plus `cat /sys/bus/pci/devices/0000:42:0{8,9}.0/current_link_speed`, once with both cameras on and once with one camera powered off.
    - Purpose: confirm `LnkCap` reports `DLActive+` and that `LnkSta` shows `DLActive-` (or the link speed changes) when a camera is off. That is the basis of the misconfiguration interlock in Requirement 2.
    - Optionally, also send `sudo dmesg | grep -i pvcam`. The driver may log BDF ↔ `pvcamPCIE_N` at probe, which would be a secondary mapping source.
    - And `cat /sys/class/pvcam_pcie/pvcamPCIE_0/uevent`, in case the virtual device carries the parent BDF.
    - Added after Q1–Q9: for stale detection and the vendor-ID interlock, run these read-only commands on the camera devices: `sudo setpci -s 0000:43:00.0 VENDOR_ID COMMAND BASE_ADDRESS_0` (and the same for `0000:44:00.0`) and `cat /sys/bus/pci/devices/0000:4{3,4}:00.0/resource | head -1`, in three conditions:
      - (i) camera on and working;
      - (ii) camera powered off;
      - (iii) camera powered back on but **not** yet hotplugged (app shut down, so nothing rescans).

      Expected: (ii) reads `ffff`; (iii) reads the real vendor ID but with COMMAND bit 1 (memory) clear and/or BAR0 different from `resource`. `setpci` without `=` only reads.

11. **Start before Q10?** May I begin implementation now (steps 1–4)? The plan would be to implement both interlock signals (link status and vendor ID `0xffff`) and the stale check behind one helper, then keep, drop, or tune them once Q10 results arrive. Recommendation: yes. Q10 affects only which config-space bits are trusted, not the structure.

Answer (2026-10-09): yes, start. Q10 to follow when the instrument is free.

## Tests
<!-- The agent should list and describe the test it plans to implement.  It should be specific about the purpose and goal of the test. -->

### Harness

- **SDK stubs:** `tests/pvcam/master.h` and `tests/pvcam/pvcam.h` hold the types, constants, and `pl_*` declarations pvcamCtrl uses. They are wired in through a `pvcamCtrl*_test` case in `tests/Makefile.one` (`CPPFLAGS += -I$(TESTS_DIR)/pvcam`), with no `-lpvcam`.
- **Fake PVCAM:** `apps/pvcamCtrl/tests/pvcamCtrl_harness.hpp` is a scripted fake library inside `\cond DOXYGEN_SUPPRESS_TEST_HARNESS`. It has a camera list (name, serial, parameter values), per-call failure injection (by function and call index), open/close/init/uninit tracking, and EOF callback capture.
- **App base:** the real app is built on `outletTestApp` (`tests/outletAppTest.hpp`) for captured logs, power callbacks, and a temp directory, as in `zaberLowLevel_power_test.cpp`. libc calls that are otherwise unreachable (`sem_init`, `sem_post`, `clock_gettime`) are redirected by macros only when pvcamCtrl.hpp is included, after libMagAOX.hpp, so libMagAOX is unaffected.
- **Fake sysfs:** a temp directory tree holds `<port>/{vendor,device,config,rescan}` and `<port>/<camera>/{vendor,device,remove}`. The PCIe helper's I/O goes through small virtual seams (attribute write, config read/write, sleep). A test subclass records the ordered operations and emulates `rescan` creating the camera child. The production seam implementations are tested separately against real temp files.
- **Structure and docs:** `libXWCTest::pvcamCtrlTest`, `\defgroup pvcamCtrl_unit_test` under `application_unit_test`, a Doxygen brief per `TEST_CASE`, and `#ifdef PVCAMCTRL_TEST_DOXYGEN_REF` blocks for real-symbol links.
- **Measurement:** `COVERAGE=1` builds and `tests/coverage/update_coverage`. The acceptance metric is 100% lines for `apps/pvcamCtrl/*.hpp`.

### Test files and cases

**`pvcamPcie_test.cpp`: PCIe helper against a fake sysfs.**
- Port validation: present with a known card ID; missing; unknown vendor:device; empty BDF (disabled).
- Camera child discovery: none, one, ignores non-camera children, ignores the sibling port's camera.
- SBR: reads BRIDGE_CONTROL, writes `|0x40`, then restores the original exactly. Includes short read and write failures.
- `remove`/`rescan` write `1` to exactly the right paths. Includes open and write failures.
- Link-status read: link active vs. inactive bit from the fake config space's PCIe capability (capability-list walk). Includes a missing PCIe capability and short reads.
- Production I/O seams against real temp files.

**`pvcamCtrl_hotplug_test.cpp`: power-off/power-on with and without cameras (the key logic).** This test file covers:

1. *App start with camera unpowered.* Power Off at startup goes to POWEROFF. No PCI or PVCAM calls are made and no errors are logged.
2. *Power on, camera appears.* Port empty → POWERON → wait → ordered ops `SBR, rescan, verify` (no `remove`) → `connect()` finds the serial → READY/OPERATING. Also checks that the lock was acquired before the first PCI op and released after.
3. *Power cycle with a stale camera.* Camera child present and app handle open → power Off → On. The handle is closed and PVCAM uninitialized **before** `remove`; the order is `SBR, remove, rescan`; then reconnect.
4. *Power on, no camera after hotplug.* NODEVICE with exactly one info log. Nothing is retried before `retryInterval`, a retry happens after it, and the camera appearing later connects.
5. *Other camera untouched.* A two-port tree with a second camera, plus a fake second open camera: no op ever names the sibling port or device, and the sibling's open state is unchanged.
6. *Misconfigured port.* The configured port's link stays up after our power-off (it is really the other camera's port): a critical log, hotplug disabled, no PCI write ever made, and the other camera is untouched.
7. *Lock contention or failure.* Another fd holds the lock, or the lock file cannot be opened: no PCI ops; retried on the next loop.
8. *Power or target goes off mid-sequence.* The sequence aborts with no error logs (power-loss suppression).
9. *sysfs write failures* at each step: error logged, NODEVICE, retried.
10. *Hotplug disabled* (no BDF): POWERON → connect exactly as today, with zero PCI ops. This guards against regressions for non-PCIe use.
11. *Startup with power on, stale camera* (fake config with Memory Space Enable clear or BAR0 mismatch): hotplug runs **before** any `pl_pvcam_init`/`pl_cam_open`, then connect. A healthy camera child (MSE set, BAR0 matches) means no hotplug and a direct connect.
12. *Startup with power on, no camera child* (host booted with camera unpowered): hotplug attempted, camera still absent, NODEVICE with exactly one diagnostic naming the reboot requirement, and retries without further log noise.
13. *Healthy child but serial not found:* the device is never reset; the app stays in NODEVICE and retries `connect()` as before.
14. *Power-on defaults* (Q4 approved): `powerOnDefaults()` and the ROI reset run on POWERON, and `onPowerOff()` invokes the dev helpers and marks hotplug pending.

**`pvcamCtrl_test.cpp`: remaining coverage of the existing app.**
- Config: missing serial number is critical; `circBuffMaxBytes`, `acqSleep`, the new `pcie.*` keys, and invalid BDF handling.
- `appStartup()`, including semaphore failures, and `appShutdown()` with and without an open handle, including uninit error vs. not-initialized.
- `connect()`:
  - zero cameras;
  - `get_total`, `get_name`, `open`, and serial-read failures;
  - camera without a serial number;
  - wrong serial;
  - match;
  - exp-res param failures;
  - fan get/set failures.
- `fillSpeedTable()` and `dumpEnum()`: every failure branch and the not-CONNECTED path.
- `getTemp()` and `getFanSpeed()`:
  - OPERATING short-circuit;
  - unavailable;
  - read failures with power on vs. power off;
  - each fan enum and an unknown value.
- stdCamera setters:
  - fan speed names, an invalid name, set failure, and both log variants;
  - exposure time min/max clamping and param failures;
  - `setFPS`;
  - `checkNextROI` clamping on all four edges;
  - `setNextROI`, `setShutter`, and the no-op setters.
- Framegrabber:
  - `configureAcquisition()` over each readout speed and the default fallback;
  - `m_fpsSetted` branches (readout-limited vs. not);
  - each `pl_*` failure;
  - buffer reallocation;
  - fan restore paths;
  - `startAcquisition`, `acquireAndCheckValid` (ready, EAGAIN, error), `loadImageIntoStream` in 8-bit and 16-bit, `reconfig`.
- EOF callback trampoline (semaphore handshake), plus the telemetry wrappers `checkRecordTimes` and `recordTelem` ×2.
- `appLogic()` flow: NOTCONNECTED → connect → OPERATING; temp and fan failure paths with power on vs. off.

## Implementation Plan
<!-- Here the agent documents its plan for the implementation -->

Branch: `jrmales/kinetix-hotplug` (current).

1. **Test infrastructure (functional commit 1)**
   - `tests/pvcam/master.h`, `tests/pvcam/pvcam.h` (stub SDK).
   - `tests/Makefile.one` pvcam case; `tests/tests.list` entries; `tests/groups.dox` `pvcamCtrl_unit_test`.
2. **PCIe helper (functional commit 2):** `apps/pvcamCtrl/pvcamPcie.hpp`, header-only, kept in the app so it counts toward the app's coverage.
   - The `pvcamPcie` class holds: sysfs root (default `/sys/bus/pci/devices`, overridable only by tests), downstream BDF, camera ID `1b6b:0001`, and the known card IDs from the vendor script.
   - Methods: `validatePort()`, `cameraDevice()`, `linkActive()`, `cameraResponds()` (vendor ID ≠ `0xffff`), `cameraStale()` (Command MSE / BAR0 vs. `resource`), `secondaryBusReset()`, `removeCamera()`, `rescan()`, `hotplug()`. `hotplug()` returns a status enum: camera present, camera absent, port missing, interlock failed, I/O error.
   - Virtual I/O seams: `writeAttr`, `readConfig16`, `writeConfig16`, `sleepFor`.
   - The helper does not log. The app logs from the returned status, which keeps log policy and power-loss suppression in pvcamCtrl.
   - Promoting this to `libMagAOX/sys` would only be worthwhile if a second PCIe-attached device needs it (follow-up).
3. **pvcamCtrl integration (functional commit 3)**
   - Config:
     - `pcie.downstreamPort` and `pcie.retryInterval` in `setupConfig`/`loadConfigImpl`;
     - validate the port in `appStartup()` (warning, not fatal, if missing);
     - members `m_pcie`, `m_hotplugPending`, `m_lastHotplugAttempt`, `m_pcieLockPath`.
   - New `pvcamCtrl::hotplugCamera()`:
     - close the handle and uninit PVCAM;
     - non-blocking `flock` on the lock file;
     - `elevatedPrivileges` scope around `m_pcie.hotplug()`;
     - release the lock;
     - log by status;
     - return 0 or a non-fatal code.
   - `connect()` takes the same lock around init and enumeration. If the lock is busy, it returns in NOTCONNECTED and retries the next loop.
   - `appLogic()`:
     - POWERON: after the wait, if hotplug is enabled and pending, run it, then NOTCONNECTED;
     - NODEVICE: retry per the requirements above.
   - If Q4 is approved:
     - reorder so `STDCAMERA_APP_LOGIC` sees POWERON;
     - add `onPowerOff()`/`whilePowerOff()` calling the dev helpers via their macros where they exist (AGENTS §18) and setting `m_hotplugPending = true`.
   - If Q6 is approved: minimal fixes to `fillSpeedTable()`, `dumpEnum()`, `getTemp()`, and the circular buffer allocation.
4. **Tests (functional commit 4, or with 2/3):** the three test files above, iterating with `COVERAGE=1` until 100%.
5. **Documentation (docs commit)**
   - Doxygen pass over all touched files: file blocks, `///` briefs, inline `/**< [in] */` params, member docs, named-section ordering. The existing `pvcamCtrl` declarations lack parameter docs on several methods (`setShutter`, `dumpEnum`, `st_endOfFrameCallback`, ...) and member docs (`m_ports`, `m_8bit`, ...).
   - New `apps/pvcamCtrl/doc/pvcamCtrl.dox` describing hotplug configuration and behavior, and `apps/pvcamCtrl/config/example.conf`, following `apps/virtualPDU`.
   - Update this plan with as-built notes.
6. **Formatting (format commit):** `clang-format` on touched files.
7. **User-run hardware validation** per Q8, then the PR. The description will carry the attribution line and point to this plan file.

### Implementation Notes (as built, 2026-10-09)

**Status:** steps 1–4 are implemented. All three suites pass. lcov line coverage: `apps/pvcamCtrl/pvcamCtrl.hpp` 734/734 and `apps/pvcamCtrl/pvcamPcie.hpp` 201/201. `pvcamCtrl.cpp` compiles against both the stubs and the real SDK headers in `/opt/pvcam/sdk/include` with `-Wall -Wextra`, with zero warnings. Q10 is still pending, so both port-down signals (link status and an all-ones vendor ID) and both stale signals (Memory Space Enable and the BAR0 mismatch) are active.

**Production changes**
- **`apps/pvcamCtrl/pvcamPcie.hpp`** (new):
  - `pvcamPcie` does the per-port SBR, remove, and rescan, plus port validation, camera classification (`pcieCamera`: none, unresponsive, stale, healthy, error), link status, and `portDown()`.
  - `pvcamPcieLock` is the non-blocking `flock` shared by all instances.
  - All I/O goes through sysfs, via virtual seams for writes, config access, and pauses.
- **`pvcamCtrl.hpp`, PCIe hotplug:**
  - `[pcie] downstreamPort, retryInterval` config;
  - port validation at startup;
  - `pcieLogic()` before `connect()` in `appLogic()`;
  - `hotplugCamera()`: lock, `releaseCamera()`, elevated `hotplug()`;
  - `checkPortDown()` in `whilePowerOff()`;
  - deduplicated diagnostics via `pcieLog()`;
  - `connect()` takes the shared lock.
- **Q4 fixes:**
  - `STDCAMERA_APP_LOGIC` now runs first, so `powerOnDefaults()` and the ROI reset run on power-on;
  - new `onPowerOff()`/`whilePowerOff()` call the stdCamera, frameGrabber, and dssShutter hooks.
- **Q6 fixes:**
  - `fillSpeedTable()`: returns `-1` on a `pl_get_enum_param` failure, reads `maxg` with `ATTR_MAX`, and uses `std::vector<char>` buffers (as does `dumpEnum()`);
  - `getTemp()`: returns `-1` instead of using uninitialized values.
- **Additional defects found and fixed:**
  - `configureAcquisition()` read `PARAM_EXPOSURE_TIME` (type `ulong64`, per the SDK header) into a `uns32`, so PVCAM wrote 8 bytes into a 4-byte local. It is now `ulong64`.
  - `loadImageIntoStream()` copied from an uninitialized pointer when `pl_exp_get_latest_frame` failed. It now returns `-1`, which the framegrabber handles by reconfiguring.
- **Behavior-neutral changes for coverage:** removed the trailing `return;` in the constructor, and gave `pvcamErrMessage()` a single return expression. Otherwise gcc attributes those closing braces only to exception cleanup.

**Design decisions made during implementation**
- **No hotplug of a healthy device unless this power-off proved the port is ours.** Resetting a port is allowed only when its camera is absent, unresponsive, or stale (so nothing usable can be disrupted), or after a power-off during which the port was seen down. This led to the two changes below.
  - **Port evidence is per power-off.** If the port comes back up while our power is still off, the app logs a CRITICAL error and disables hotplug, since the port must belong to another camera. An earlier idea of a lifetime "verified" flag was rejected: it could be satisfied while both cameras were off and later allow resetting the other camera.
  - **The serial-not-found fallback was dropped** (Requirement 4, test 13), because it would reset a healthy device on an unverified port.
- **Lock file errors:** if the lock file cannot be opened, hotplug is refused (logged once), but `connect()` proceeds unlocked so camera operation is preserved. Both instances share the path, so neither can hotplug in that state.
- **Hotplug retries:** `m_hotplugPending` stays set until a camera is found. Every attempt, including the first after power-on (`m_lastHotplug` is reset in `onPowerOff()`), is rate limited by `retryInterval`.

**Test harness** (`apps/pvcamCtrl/tests/pvcamCtrl_harness.hpp`):
- SDK stubs in `tests/pvcam/{master.h,pvcam.h}` are added to the include path only for `pvcam*_test` (`tests/Makefile.one`).
- A scripted fake PVCAM library with per-call fault injection.
- Macro redirection for:
  - `MagAOXApp`: `pvcamTestApp`, with a placeholder framegrabber thread and a state hook used to simulate a concurrent power change;
  - `telemeter`: `outletTestTelemeter`;
  - `dssShutter`: a threadless double with injectable results;
  - `pvcamPcie`: `pvcamTestPcie`, which records operations and emulates the kernel's remove and rescan;
  - `sem_init`, `sem_trywait`, `sem_post`, and `clock_gettime`.
- A fake sysfs tree with config-space images.
- The circular-buffer allocation failure is exercised with `RLIMIT_AS`, with no production seam.

**Documentation (step 5):**
- Doxygen pass on `pvcamCtrl.hpp`: briefs, inline parameter and return docs, member docs, and group titles. The `setFPS()` and `checkNextROI()` briefs were corrected.
- Briefs added to the stub SDK declarations.
- New `apps/pvcamCtrl/doc/pvcamCtrl.dox` and `apps/pvcamCtrl/config/example.conf`.
- `PVCAMCTRL_TEST_DOXYGEN_REF` added to `PREDEFINED` in `doc/config/Doxyfile.libMagAOX`, so the test reference blocks link to the real methods (verified with a restricted Doxygen run).

**Suites**
- `pvcamPcie_test`: 6 cases;
- `pvcamCtrl_hotplug_test`: 9 cases, covering the key power-off/power-on scenarios with and without cameras;
- `pvcamCtrl_test`: 10 cases.

## Follow-up and Edge Cases
<!-- The agent should list any planned follow up and any edge cases that are not addressed -->

- **Simultaneous power-on of both cameras:** the lock serializes the two hotplugs. Each SBR touches only its own port, so ordering does not matter.
- **Camera present at boot, never power-cycled, app restarted:** no hotplug runs unless `connect()` fails to find the serial (Req. 4), so a live camera is not reset needlessly.
- **Unseen power cycle while the app was down:** covered by stale detection (Requirement 4). How `pl_cam_open` behaves on a stale device that the check misses is still unknown; validation case (d) exercises this.
- **Host booted with a camera unpowered:** not recoverable by hotplug, per the user. The app reports it and waits; recovery requires a host reboot with the camera powered.
- **Slot or BDF changes after hardware maintenance:** the startup validation logs this clearly, but the config must be updated by hand.
- **The other app's PVCAM state after our rescan:** the other process's camera is never touched, and it re-inits PVCAM only on its own reconnect. No action expected; confirm in hardware validation.
- **Generalization:** if other PCIe-attached devices need hotplug, move `pvcamPcie` into `libMagAOX/sys` with its own tests.
- **Vendor script drift:** card IDs and timings are copied from the script as of 2026-10-09. A future PVCAM release could change them. Record the source version in the helper docs.

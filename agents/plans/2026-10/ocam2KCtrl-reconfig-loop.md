# ocam2KCtrl configuration loop

Started on 2026-10-31 on workstation `mx11`, following the report "ocam2KCtrl seems
to have become prone to infinite loops of reconfigure. Has been triggered a couple
of times today by enabling syncro." No hardware commands were run. The user
supplied the camwfs log. Changes are recorded on `jrmales/ocam-reconfig-loop`,
built on `jrmales/dm-syncopation` (`5baea984`) for a PR back into that branch.

## Evidence

camwfs was running `63938bd` (`jrmales/dm-syncopation` merged with `dev`). The log
shows repeated passes about 2.1 s apart: `camera configured with: mode 3`,
`set fps: 0`, `set synchro: on|off`, `OCAM2K initialized`. The loop started when
synchro was enabled at 23:35:50 and continued after synchro was disabled.

- No looping pass has `Loading EDT PDV config file`, which `pdvConfig()` logs on
  every `reconfig()`. The worker was not consuming `m_reconfig`.
- No `sleep(1)` retry, and no error, skip, or corrupt-frame logs.

So the grab loop exited without a recovery request, and the framegrabber reran
`configureAcquisition()` without resetting the EDT device. An earlier loop
(21:47:18–21:47:47) ended right after a real `reconfig()`.

Repeated `INDI property still unresolved after retry backoff: pdu1.camwfs` notes
point to a power target that is not `On`. The outlet `target` element starts empty
and only becomes `On` when an outlet is switched on, so a PDU restart while camwfs
is powered leaves `powerStateTarget() == -1`. This still needs to be confirmed
from `pdu1.camwfs.target` on the instrument.

## Cause

`acquireAndCheckValid()` returned `-1` without requesting recovery whenever
`powerState() != 1 || powerStateTarget() != 1`. The intent was to stay quiet
during power-off. An unknown target also matched, so any failed acquisition broke
the grab loop silently. The next pass re-sent the camera mode and synchro commands
to an EDT device that still had a failed acquisition, the first frame failed
again, and the cycle repeated. The corrupt-frame-number branch had the same
condition.

The framegrabber's atomic reconfiguration change (see
[frameGrabber-reconfiguration-race.md](frameGrabber-reconfiguration-race.md)) is
not involved. An earlier attempt on this branch that removed the sync-frequency and
connect-path reconfiguration requests was dropped. Neither request appears in the
loop, and on `dm-syncopation` OCAM housekeeping relies on `m_reconfig` to defer
serial queries.

## Fix

- Both branches skip recovery only when power is known to be off or switching off
  (`powerState() == 0 || powerStateTarget() == 0`), matching `powerOffRequested()`
  in the zaber apps. Otherwise they request recovery, which runs `pdvConfig()`.
- A failed SDK acquisition now logs `acquisition failed, requesting recovery` at
  `LOG_ERROR`, so this path is visible in the logs.

## Verification

New test case: "ocam2KCtrl acquisition failures request recovery with an unknown
power target". Against the unfixed header it fails in both sections (5 of 11
assertions). With the fix:

| Suite | Assertions | Cases | Result |
| --- | ---: | ---: | --- |
| ocam2KCtrl_test | 902 | 36 | Passed |
| ocam2KCtrl_reconfig_test | 379 | 4 | Passed |
| ocam2KCtrl_housekeeping_test | 640 | 26 | Passed |
| ocam2KCtrl_lifecycle_test | 7 | 1 | Passed |
| ocam2KCtrl_emgain_test | 34 | 1 | Passed |
| ocam2KCtrl_temperature_test | 504 | 5 | Passed |

## Follow-ups

- Confirm `pdu1.camwfs.target` on the instrument. If it is `On`, the remaining
  silent exit is `startAcquisition()` failing on `pdv_timeouts() < 0`.
- The framegrabber reruns `configureAcquisition()` after any `-1` from
  `acquireAndCheckValid()` without cleanup. For EDT cameras, that may warrant a
  generic cleanup request.
- `powerStateTarget() != 1` is used across many apps to mean "powering off". An
  unknown target after a PDU restart weakens all of them. A shared
  `MagAOXApp::powerOffRequested()` (generalizing the zaber helpers), or
  publishing a definite outlet target, would address this.

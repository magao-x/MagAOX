# mfilt copy (ddspc_mode mfilt_copy_8k and mfilt_copy_2k)

2026-11-25. Branch `parkertjohnson1/accel_ddspc`. Workstation `ASTR-LT-428.local`; no instrument access.

## Why

User direction (2026-11-25): a GUI-selectable mode that takes the main loop's tip/tilt mfilt, reproduces it exactly,
and applies it on `dm01disp05` while the main loop's tip/tilt input is zero. No other mode may change. It isolates the
own-channel path (modeCube shapes and sign, synco timing, `dm01disp05` routing) from DDSPC: if it holds the loop as
the main loop does, the path is right; if it does not, the problem is there and not in the model.

User direction (2026-11-25, follow-up): two separately selectable modes doing the same copy: `mfilt_copy_8k` writes it
on every synco slot (8 kHz), `mfilt_copy_2k` writes it once per WFS frame (one command every four slots).

## Design

- `ddspc_mode` gains `mfilt_copy_8k` and `mfilt_copy_2k` (first committed as a single `mfilt_copy`, renamed before
  push). Internally the mode is `ShadowMode::Standard` with `m_mfiltCopy`, so the standard
  path, its slot plan, synco output, interlocks, faults and missed-frame fill are reused unchanged.
- Per WFS frame with `predict` on: the plan's delta is the measurement itself (no `gain`, no model, no probes), and the
  existing standard integrator applies `s = clamp(mult*s - gain*y, +/-limit)` with the main loop's parameters
  (`holoop` scalars times `aol1_m*fact`, as for `[integrator]`). `s` is held on the frame's four slots and written
  through `modeCube` to `dm.channelName`. It integrates even with `[integrator] enabled=false`.
- `mfilt_copy_8k`: as standard mode, every slot renders the held `s`, so synco writes `dm01disp05_aux` at 8 kHz with
  the value changing at 2 kHz.
- `mfilt_copy_2k` (`m_copyOnce`): `outputCandidate()` offers each decision (`m_planDecisions`, counted where the plan is
  stored) only until synco has copied it. Declining a slot makes synco skip it without a fault and hold the last copy on
  the DM, so the channel is written once per WFS frame. The copy receipt (`copied()`, already under `m_commandMutex`)
  marks the decision written only if its image equals the image offered for it, so a late receipt for the previous
  decision cannot mark a newer one; an offered slot that is not copied (late) is retried on the next slot. Declined
  slots are counted in `ddspc_performance.transport_skipped` (three per frame), so that counter is not a health measure
  in this mode. The output's own publication count was not used: `copied()` runs under the output's lock, and reading
  it from the candidate would add an output-lock acquisition to every slot.
- Modes 0 and 1 of the main-loop modal output are zero from the accepted `predict` until the own-channel zero is
  confirmed, exactly as in standard mode, so mfilt's tip/tilt state holds (or leaks with mc < 1) on its own channel and
  the copy adds to it from zero: no jump at handover.
- Nothing learns: standard training is off in this mode, `learn_std` is refused ("mfilt_copy runs no DDSPC model"),
  probes are never added. Prediction needs the integrator parameters, not a ready model. A model numerical fault does
  not inhibit the copy.
- `predict` turns the copy on and off; off zeroes the channel and returns tip/tilt to the main loop as in standard mode.
  Mode changes keep the existing interlock (prediction off, zero confirmed).
- Status: `requested_mode=mfilt_copy_8k` or `mfilt_copy_2k`; `prediction` is `active` once synco acknowledges a copy, never `warming`.

Differences from the real mfilt that remain by construction: the copy is written at the next DM slot through synco
rather than by mfilt's own DM write, and the shape is `modeCube` planes 0 and 1 rather than cacao's own modes-to-DM
conversion. Those are what the mode tests.

## Verification (offline, development Mac)

- `check-rtc-controller` 37/37. 2k case: each of 12 frames offers exactly one slot, equal to the reference mfilt, and
  declines the other three; an uncopied (late) slot is offered again on the next; a late receipt for the previous
  decision does not stop the newer decision being offered. 8k case: refused until the main loop's parameters arrive; then, with `[integrator]`
  off, learning on and `gain=0.2`, every slot equals the reference `clamp(0.9*s - g_i*y, +/-limit_i)` (per-mode gain and
  limit factors, limit reached); no standard updates; `learn_std` refused; `main_loop_tt=zeroed`,
  `requested_mode=mfilt_copy_8k`; off returns the channel and state to zero.
- `check-operational-controls` 16/16 and config tests OK.
- T1 `standard_equivalence.py --base HEAD`: 192000 standard rendered values bit-for-bit identical.

## Late render for standard-path slots (after the first on-sky test)

The first on-sky `mfilt_copy_2k` test diverged. The DM worker (`SyncoOutput`) asks for a slot's image as soon as the
slot is selected, about a slot before its cutoff, and the producer publishes it at `cutoff - writeLeadUs`. The assisted
candidate already waited until `cutoff - dm.renderLeadUs` before calculating; the standard candidate (standard DDSPC
and both mfilt copies) rendered at once. A WFS decision completed between selection and render therefore reached the DM
a slot (125 us) later than necessary, adding up to a quarter frame of delay relative to the main loop.

The wait is now `renderWait()`, shared by both candidates: sleep to 20 us before `cutoff - renderLeadUs`, then poll the
clock. The rendered values for a given decision and slot are unchanged (T1 bit-for-bit); only which decision a slot
carries changes, to the newest. Latency is not yet measured on the RTC; `margin_us_*` covers only the assisted path.

Verification: `check-rtc-controller` 38/38. The new case selects a slot 5 ms before its cutoff, completes a decision
while it waits, and requires the slot rendered no earlier than `cutoff - renderLeadUs` with the newer decision; without
`renderWait()` in the standard candidate it fails. `check-operational-controls` OK. T1 192000 values identical.

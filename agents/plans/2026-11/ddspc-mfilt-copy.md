# mfilt copy (ddspc_mode mfilt_copy)

2026-11-25. Branch `parkertjohnson1/accel_ddspc`. Workstation `ASTR-LT-428.local`; no instrument access.

## Why

User direction (2026-11-25): a GUI-selectable mode that takes the main loop's tip/tilt mfilt, reproduces it exactly,
and applies it on `dm01disp05` while the main loop's tip/tilt input is zero. No other mode may change. It isolates the
own-channel path (modeCube shapes and sign, synco timing, `dm01disp05` routing) from DDSPC: if it holds the loop as
the main loop does, the path is right; if it does not, the problem is there and not in the model.

## Design

- `ddspc_mode` gains `mfilt_copy`. Internally the mode is `ShadowMode::Standard` with `m_mfiltCopy`, so the standard
  path, its slot plan, synco output, interlocks, faults and missed-frame fill are reused unchanged.
- Per WFS frame with `predict` on: the plan's delta is the measurement itself (no `gain`, no model, no probes), and the
  existing standard integrator applies `s = clamp(mult*s - gain*y, +/-limit)` with the main loop's parameters
  (`holoop` scalars times `aol1_m*fact`, as for `[integrator]`). `s` is held on the frame's four slots and written
  through `modeCube` to `dm.channelName`. It integrates even with `[integrator] enabled=false`.
- Modes 0 and 1 of the main-loop modal output are zero from the accepted `predict` until the own-channel zero is
  confirmed, exactly as in standard mode, so mfilt's tip/tilt state holds (or leaks with mc < 1) on its own channel and
  the copy adds to it from zero: no jump at handover.
- Nothing learns: standard training is off in this mode, `learn_std` is refused ("mfilt_copy runs no DDSPC model"),
  probes are never added. Prediction needs the integrator parameters, not a ready model. A model numerical fault does
  not inhibit the copy.
- `predict` turns the copy on and off; off zeroes the channel and returns tip/tilt to the main loop as in standard mode.
  Mode changes keep the existing interlock (prediction off, zero confirmed).
- Status: `requested_mode=mfilt_copy`; `prediction` is `active` once synco acknowledges a copy, never `warming`.

Differences from the real mfilt that remain by construction: the copy is written at the next DM slot through synco
rather than by mfilt's own DM write, and the shape is `modeCube` planes 0 and 1 rather than cacao's own modes-to-DM
conversion. Those are what the mode tests.

## Verification (offline, development Mac)

- `check-rtc-controller` 36/36. New case: refused until the main loop's parameters arrive; then, with `[integrator]`
  off, learning on and `gain=0.2`, every slot equals the reference `clamp(0.9*s - g_i*y, +/-limit_i)` (per-mode gain and
  limit factors, limit reached); no standard updates; `learn_std` refused; `main_loop_tt=zeroed`,
  `requested_mode=mfilt_copy`; off returns the channel and state to zero.
- `check-operational-controls` 16/16 and config tests OK.
- T1 `standard_equivalence.py --base HEAD`: 192000 standard rendered values bit-for-bit identical.

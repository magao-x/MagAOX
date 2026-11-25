# DDSPC mfilt clone modes (mfilt_clone_8k, mfilt_clone_2k)

## Purpose

The first on-sky `mfilt_copy_2k` test diverged. The request (with the guidance quoted from the team): a test "as close
to a clone of the main loop as possible, with the only difference coming from writing to the DM through the synco, with
a verification that the TT is written in time to be in sync with the main loop write (step 0)", using
`aol1_modevalDM`, mfilt's output, "so when you switch you pick up right where it left off". Two new `ddspc_mode`
elements, toggled in loPredCtrlAccGUI. No other mode changes.

## Design

Both are the mfilt copy (`agents/plans/2026-11/ddspc-mfilt-copy.md`) with three differences:

- **Start from mfilt.** At the first decision after `predict` is accepted, the clone's state `s` is set to modes 0 and 1
  of `aolN_modevalDM` (`[loop] number`; opened by the accepted `predict` request, which is refused if the stream is not
  FLOAT with those modes). The two floats are read between unchanged `cnt0` with `write` clear; a torn read leaves the
  frame at zero and retries on the next. The values read are published as `mfilt_clone.start_tt0/1`.
- **No doubling.** From that frame mfilt's tip/tilt input is zero (as in every own-channel mode), so mfilt keeps its last
  output on its own DM channel, decaying by `mult` each frame. The clone models that held output `h` (`h = mult*h` each
  frame, from the value read) and writes `s - h` on `dm.channelName`, so the DM total is `s`. `s` follows mfilt's law,
  `s = clamp(mult*s - gain*y, +/-limit)`, with the limit on the whole state as in mfilt (the copy limits only its own
  part). Without the subtraction, starting `s` from mfilt's output would put the full tip/tilt correction on the DM
  twice at the switch.
- **Step 0.** `mfilt_clone_2k` offers its command only to step 0, the slot synco copies just before the main loop's write
  (`agents/plans/2026-10/dm-syncopation.md`). A decision not ready by step 0's render waits for the next step 0, a frame
  later, and synco holds the channel on steps 1-3. `mfilt_clone_8k` writes every slot like `mfilt_copy_8k`.

Slots are rendered at `cutoff - dm.renderLeadUs` (the late render added for the standard path), so the newest decision is
used.

## Step-0 verification

Each step-0 request in either clone mode is classified before rendering:

- **in sync**: the stored decision is for the newest WFS frame `processImage` has received, and that frame arrived after
  the previous step 0's target (the previous main-loop write). It is then the frame this cycle's main-loop write
  carries. The margin, render time minus decision completion, is recorded.
- **missed**: otherwise, either a newer frame has arrived and is not yet decided, or the decided frame arrived before the
  previous write, which already carried it. The clone's command is then a frame behind mfilt's.

The first step 0 after `predict` only records its target. INDI `mfilt_clone`: `step0_in_sync`, `step0_missed`,
`step0_margin_us_min`, `step0_margin_us_mean` (min/mean since the last appLogic update), `start_tt0`, `start_tt1`.

Assumptions: step 0's `targetNs` marks the main-loop write; a frame that arrives in shared memory while the previous frame
is still being processed is not yet counted as received (only when processing is slower than a frame).

## Limits

- `h` is modelled. If mfilt's held output does not decay as `mult*h` (for example if `aolN_modevalDMf` feedback rewrites
  its state), the DM total differs from `s` by that difference. `aolN_modevalDM` is read only at the start, to add no
  per-frame reader to a loop stream.
- If mfilt has already processed the first frame when the clone reads `aolN_modevalDM`, the start is `mult*h` rather than
  `h`: a one-time error of `(1 - mult)*h`.

## Verification (offline, development Mac)

- `check-rtc-controller` 40/40. 8k: refused without the stream; a torn read gives a zero frame and no start; then every
  slot equals `s - h` of the reference (start .2/-.1, per-mode gain and limit factors, limit reached on `s`); no fit
  updates; `requested_mode=mfilt_clone_8k`, `main_loop_tt=zeroed`, `start_tt0/1` published; off returns to zero and
  re-reads on the next start. 2k: steps 1-3 never offered; five frames decided before their step 0 are written there
  and counted in sync; an undecided newer frame and a decision for an already-carried frame are counted missed;
  INDI counts published.
- `check-operational-controls` OK. T1 `standard_equivalence.py --base HEAD`: 192000 standard values bit-for-bit.

# ACC-assisted DDSPC at 2 kHz with 8 kHz averaged ACC samples

## Purpose

The request: "We want another mode to toggle on the gui that allows us to average the past 4 accel measurements on the
pi when it is running at 8khz, then we send a stable 2kHz acc data stream for loPredCtrlAcc to use to populate the acc
history vector in acc assisted ddspc. [...] It should be nearly the same as acc assisted at 2kHz with the only change
being the rolling average of the previous 4 measurements (when acc at 8kHz) so that there is a stable 2kHz average acc
value we use".

## Design

- **mcp3208Ctrl** (branch `parkertjohnson1/mcp3208Ctrl`): `accel.average` / INDI `average` (1-16, default 1). Each
  published frame is the rounded mean of N consecutive scans, so `acc000` runs at `fps/N`; `fps()` (the framegrabber's
  rate) and a new read-only `stream_fps` report it. At 8 kHz with N = 4 the stream is 2 kHz of 4-scan means. An
  incomplete average is never published (reconfiguration or shutdown mid-average). The frame keeps the newest scan's
  timestamp, (N-1)/2 scans (187.5 us at 8 kHz, N = 4) after the mean's centre. The stream stays UINT16 (the mean is
  rounded), so every consumer's calibration is unchanged; averaging 4 scans halves white noise but the rounding keeps
  the 12-bit step. In synchronized (semaphore) timing the 4 scans of a WFS frame average to one sample per frame.
- **loPredCtrlAcc** (this branch): no change. `accel_assisted_2k` (`agents/plans/2026-11/ddspc-assisted-2k.md`) is the
  controller mode, and arrival-order detection (`agents/plans/2026-11/ddspc-detected-rates.md`) sees one ACC sample per
  WFS frame, `acc_ratio = 1`; the window is the newest `accel_history` averaged samples.
- **loPredCtrlAccGUI** (branch `parkertjohnson1/loPredCtrlAcc_gui`): an "ACC Assisted 2 kHz (8 kHz avg)" mode button
  sends `acc000.fps = 8000`, `acc000.average = 4`, then `ddspc_mode.accel_assisted_2k`. It is offered with the other
  mode buttons (prediction off, zero confirmed) and only when the accelerometer publishes `average` and `fps`
  (mcp3208Ctrl, not accelSim). Every other mode button sets `average` back to 1 if it is averaging. The button is shown
  selected when `accel_assisted_2k` is on and `average > 1` (plain ACC Assisted 2 kHz otherwise); an "ACC averaging" row
  and the banner show the averaging.

## Operation

The rate change resets the models through the detected `acc_ratio` (4 or 1); retrain after switching. A consumer that
sizes PSDs from `acc000.fps` (modalPSDs `circBuff.fpsProperty`) should use `stream_fps` while averaging.

## Verification (offline, development Mac)

- mcp3208Ctrl: tests "publishes the rounded mean of average scans" (four 8 kHz scans -> one frame of their rounded
  means, fps() 2000; average 1 publishes every scan) and "average callback sets the scans per published frame"; g++-15
  syntax check only on the Mac, to be built and run on the Pi.
- loPredCtrlAccGUI: test "selects ACC-assisted 2 kHz with 8 kHz averaged samples" (offered only with `average`; the
  three commands; selected state, status row and banner from the readbacks; another mode sends `average = 1`); 14/14
  panel tests pass.

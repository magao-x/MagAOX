dmTemporalResponse
==================

[TOC]

------------------------------------------------------------------------

# NAME

dmTemporalResponse - measure the temporal step response of a DM with sub-frame resolution using camWFS.

# SYNOPSIS

```
dmTemporalResponse -n dmTemporalResponse
```

Runs are started and controlled from cursesINDI (see [INDI](#indi)).

# DESCRIPTION

`dmTemporalResponse` measures how a deformable mirror responds in time to a step command, sampled faster than the
camWFS frame rate.

For every trial the app:

1. waits for the camWFS frame semaphore (the *trigger* frame);
2. busy-waits until `atime + d`, where `atime` is the trigger frame's acquisition time and `d` is the delay;
3. writes the poke command to the DM channel and records the time it was written;
4. captures the next N camWFS frames;
5. zeroes the DM and waits the settle time.

The delay `d` steps through an evenly spaced grid `d_k = k * span / K`, `k = 0..K-1`.  By default the span is one
camWFS frame period, so the K delays sample the step response at K phases within a frame.  For each delay the app
runs M/2 trials with a positive poke followed by M/2 trials with a negative poke, and saves the averaged half
difference `(mean(+) - mean(-)) / 2`.  The +/- difference removes the static WFS signal and the camera bias, so no
dark is needed.

A reference pass (`nRef` +/- pairs at `d = 0`) runs first.  If the reference pattern shows no real WFS response
(peak SNR below `analysis.minRefSNR`, see below), the run fails with "no WFS response to poke" instead of producing
meaningless curves; `reference.fits` is still written for diagnosis.  Its steady-state difference image `P`, averaged over the
last `nSettle` frames, is used to reduce each frame to a scalar response `r` (0 unpoked, 1 fully settled).  From the
per-trial curves the app computes, per delay:

| Metric | Definition |
|---|---|
| `t50` | time from the DM command to `r = 0.5` [us] |
| `rise` | 10-90% rise time [us] |
| `overshoot` | `max(r) - 1` |
| `settleErr` | RMS of `r - 1` over the last `nSettle` frames |
| `jitter` | trial-to-trial std of `r` at the frame nearest `t50` |
| `delayErrMean`, `delayErrStd` | achieved minus requested delay [us] |
| `lateFrac` | fraction of trials whose poke deadline had already passed when the frame was received |

It also combines all delays on the time-since-command axis into a super-sampled response, binned at
`frame period / resampleFactor`, and reports the best delay by the chosen criterion.

## Poke modes

- `actuator` (default): exactly one actuator `(x, y)`; the command is `+/- amp` at that actuator.
- `pattern`: a 2-D FITS file on the RTC with the same size as the DM channel; the command is `+/- amp * pattern`
  (not normalized, so `amp = 1` applies the file as stored).  The file is re-read at each run start, and a copy is
  saved with the results.

- `t2wMode`: plane `t2w.mode` (default 4, astig00) of t2wOffloader's woofer mode cube `t2w.wooferModesFile`
  (`wModes.fits`); the command is `+/- amp * plane`.  This is the woofer side of a single-mode offload.  The file is
  re-read at each run start and its SHA-256 recorded.

In every mode the largest absolute command must not exceed `poke.maxCommand` (default 1).

## Counter-offload cases

`counter.case` (or INDI `offload_case`) selects what each measurement trial does.  The app then stands in for
t2wOffloader so that counterOffload makes its real counter ramp, and the WFS response is measured with the loop open.

| Case | Trial |
|---|---|
| `woofer` (default) | the woofer poke only, as above |
| `counter` | no woofer poke; `counter.postDelayUs` after the time it would have been written, post `+/- amp * Δ` to `counter.stream` |
| `both` | the woofer poke, then `counter.postDelayUs` later post `+/- amp * Δ` |

- **Δ** is plane `t2w.mode` of t2wOffloader's tweeter mode cube `t2w.tweeterModesFile` (`tModesOrtho.fits`).
  t2wOffloader offloads Σ δₚ `wModes[p]` to the woofer and posts Σ δₚ `tModesOrtho[p]` as its tweeter offload, so a
  woofer poke of `amp * wModes[k]` and a Δ of `amp * tModesOrtho[k]` are exactly one single-mode offload.  Both cubes
  are written by t2wOffloader at startup in `/tmp`; copy them to a permanent place.
- **The post.**
  - Δ is written to `counter.stream` (default `aol0_tweeterOffload`, the stream counterOffload watches) with
    `ImageStreamIO_UpdateIm_atime`.
  - Its `atime` is the `CLOCK_ISIO` time of the woofer write, as t2wOffloader stamps it.
  - counterOffload places its ramp from when it receives Δ, so the delay between the woofer write and the post shifts
    the counter directly.
  - `counter.postDelayUs` < 0 (the default) measures t2wOffloader's own gap at run start, as `writetime − atime` of
    its last post still in the stream.
  - The value used is recorded as `POSTDLY`, and the achieved delay per cube as `POSTMEAN`/`POSTSTD`.
- **Zeroing.** After each trial that posted Δ the app requests `counterOffload.zero`, which clears counterOffload
  exactly whatever its leak or clipping.  `poke.settle` must cover the INDI round trip; 0.05 s is ample.
- **Reference.** The reference pass always pokes the woofer, so every case is projected onto the woofer's pattern `P`
  and is in units of a full woofer step:
  - the woofer settles at 1;
  - the counter at −gain × (counter/woofer scale at the WFS);
  - `both` at the residual, 1 minus that.
- **Time origin.** All curves are timed from the woofer write (in `counter`, when it would have been), so the
  counter's curve includes its whole latency: the post delay, counterOffload's handoff and slot wait, the synco lead,
  dmcomb, the driver, and the tweeter mechanics.
- **Metrics.** In `counter` the timing metrics (`t50`, `rise`, `jitter`) are taken on each curve normalized by its own
  settled value.  In `both` they are usually unavailable, and the residual is described by `settled`, `peakAbs` and
  `residRms`.
- **Start checks.** For `counter` and `both` a run refuses to start, and logs why, unless:
  - the poke mode is `t2wMode`;
  - t2wOffloader's `offload` is reported off, since the app writes the stream t2wOffloader posts to;
  - counterOffload's `enable` is reported on;
  - holoop's `loop_state` is reported off;
  - `counter.stream` is a 2-D float image the size of the tweeter plane;
  - the post delay is known.

  counterOffload's `rampStartUs`, `rampDurationUs`, `gain` and `leak` are recorded in the headers. If any changes
  during the run, `COCHG = 1` is recorded.

## Timing notes

- The trigger, delay busy-wait, and poke run on the shmimMonitor thread.  Set `wfscam.threadPrio` and
  `wfscam.cpuset` appropriately on the RTC for accurate delays.
- Delays shorter than the time it takes the app to receive the frame are *late*: the poke is written immediately and
  the achieved delay is recorded.  If more than `maxLateFrac` of the trials at a delay are late, a warning is logged
  with the minimum achievable delay and the cube is flagged (`LATEFLAG = 1`).
- Delays longer than one frame period are supported.  A deadline is deferred to a later frame only if it is at least
  one frame period after the current frame is processed; otherwise the poke is made by busy-waiting from the current
  frame.  So only delays shorter than the receipt latency are late, and a `delaySpan` of two frame periods (with a
  step no larger than the super-sampling bin) gives data in every super-sampled time bin.  For a delay between one
  frame period and one frame period plus the receipt latency, the first captured frame ends before the poke and has a
  negative time from the command.
- The measured response includes the full DM path (dmcomb channel sum, driver, and mechanics).
- The app requires `atime` to be `CLOCK_REALTIME`, and refuses to start if the most recent frame's `atime` is more than
  1 s from the current time.
- The DM channel is opened non-passive, so each poke increments the stream's frame counter and dmcomb applies it.
- A run aborts if camWFS changes fps or is re-allocated, if a trial times out, or if frame-counter gaps exceed
  `maxRetries` at one delay.  The DM channel is always returned to zero.

# CONFIGURATION

| Key | Type | Default | Description |
|---|---|---|---|
| `wfscam.shmimName` | string | `camwfs` | camWFS stream |
| `wfscam.camDevName` | string | = shmimName | INDI device used to read `fps` and `emgain` |
| `holoop.devName` | string | `holoop` | INDI device whose `loop_state` is recorded in the headers |
| `wfscam.threadPrio` | int | 2 | real-time priority of the trigger/poke thread |
| `wfscam.cpuset` | string | | cpuset of the trigger/poke thread |
| `dm.index` | int | 0 | DM number `NN` in `dm<NN>disp<MM>`: 0 woofer, 1 tweeter, 2 NCPC |
| `dm.channel` | int | 7 | dmcomb channel `MM` in `dm<NN>disp<MM>` |
| `poke.mode` | string | `actuator` | `actuator` or `pattern` |
| `poke.x`, `poke.y` | vector<int> | | the actuator, exactly one entry each |
| `poke.patternFile` | string | | FITS pattern path (pattern mode) |
| `poke.amp` | float | 0 | poke amplitude; must be non-zero |
| `poke.maxCommand` | float | 1 | largest absolute command allowed |
| `poke.nDelays` | int | 20 | K; with the default span, the delay step is one super-sampling bin |
| `poke.delaySpan` | float | 0 | span of the delay grid [us]; `<= 0` means two camWFS frame periods, which fills every super-sampling bin |
| `poke.nFrames` | int | 20 | N, frames per trial |
| `poke.nTrials` | int | 20 | M, trials per delay, must be even |
| `poke.settle` | float | 0.05 | seconds between zeroing the DM and the next trial |
| `poke.trialTimeout` | float | 2 | seconds to wait for one trial |
| `poke.maxRetries` | int | 5 | invalid-trial retries per delay |
| `analysis.nRef` | int | 10 | reference-pass +/- pairs |
| `analysis.nSettle` | int | 10 | trailing frames for `P` and `settleErr`; must be >= 2 (used to estimate the noise of `P`) |
| `analysis.maskThresh` | float | 0.1 | pixel mask threshold, fraction of `max|P|` |
| `analysis.resampleFactor` | int | 10 | super-sampling factor |
| `analysis.bestMetric` | string | `jitter` | `jitter`, `rise`, or `t50` |
| `analysis.maxLateFrac` | float | 0.1 | late-poke warning threshold |
| `analysis.minRefSNR` | float | 8 | minimum reference peak SNR, max\|P\| / sigmaP; pure noise gives ~4.4 for 120x120 |
| `t2w.mode` | int | 4 | mode index k in t2wOffloader's cubes (`t2wMode`) |
| `t2w.wooferModesFile` | string | | t2wOffloader's woofer mode cube (`wModes.fits`) |
| `t2w.tweeterModesFile` | string | | t2wOffloader's tweeter mode cube (`tModesOrtho.fits`) |
| `counter.case` | string | `woofer` | `woofer`, `counter`, or `both` |
| `counter.stream` | string | `aol0_tweeterOffload` | stream written with Δ |
| `counter.postDelayUs` | float | -1 | woofer write to Δ post [us]; < 0 measures t2wOffloader's gap |
| `counter.device` | string | `counterOffload` | counterOffload INDI device |
| `counter.offloadDevice` | string | `t2wOffloader` | t2wOffloader INDI device |
| `output.baseDir` | string | `/home/xsup/dm_response` | root of the output tree |

Example `dmTemporalResponse.conf`:

```
[wfscam]
shmimName=camwfs
threadPrio=60

[dm]
index=0
channel=7

[poke]
mode=actuator
x=5
y=5
amp=0.05
nDelays=10
nFrames=20
nTrials=20
```

# INDI

Tunable properties (`current`/`target`), all rejected while a run is in progress:
`dm_index`, `dm_channel`, `poke_mode` (selection: `actuator`/`pattern`/`t2wMode`), `poke_x`, `poke_y`, `pattern_file`,
`poke_amp`, `nDelays`, `delaySpan`, `nFrames`, `nTrials`, `settle`, `nSettle`, `maskThresh`, `bestMetric`,
`offload_case` (selection: `woofer`/`counter`/`both`), `t2w_mode`.  The two mode cube paths are config only.

Subscribed: t2wOffloader `offload`; counterOffload `enable`, `rampStartUs`, `rampDurationUs`, `gain`, `leak`.

Controls: `start.request`, `stop.request`.

Read-only:

| Property | Elements |
|---|---|
| `dm_stream` | `name`: the resolved stream, e.g. `dm00disp07` |
| `run_state` | `status` (idle/running/done/error/stopped), `phase` (none/reference/measuring/analyzing) |
| `progress` | `delay_index`, `delay_us`, `sign`, `trial`, `n_invalid`, `late_frac` |
| `delays` | `values`: the delay grid, comma separated |
| `results` | `t50`, `rise`, `jitter`: per-delay values, comma separated |
| `best` | `delay_us`, `t50`, `rise`, `jitter` |
| `pattern_info` | `info`: loaded pattern path, size, and SHA-256, or the validation error |
| `output` | `dir`: the run directory |

# OUTPUT

Each run writes to `<output.baseDir>/<YYYY-MM-DDTHHMMSS>/` (UTC):

| File | Contents |
|---|---|
| `dmresp_delay_<DDDDD>us.fits` | one per delay: `[nx, ny, N]` averaged +/- half-difference cube |
| `reference.fits` | `[nx, ny, 2]`: plane 0 `P`, plane 1 the mask; header `REFSNR`, `REFSIGMA`, `MINRSNR` |
| `summary_curves.fits` | `[N+1, K, 3]`: mean response, trial std, and mean time from the command [us]; row 0 is the pre-poke baseline (r = 0) |
| `summary_metrics.fits` | `[K, 12]`: delay, t50, rise, overshoot, settleErr, jitter, delayErrMean, delayErrStd, lateFrac, settled, peakAbs, residRms (columns named by `MCOLn`) |
| `summary_superres.fits` | `[nBins, 2]`: bin time [us] and binned response |
| `pattern.fits` | pattern mode only: copy of the applied pattern |
| `woofer_poke.fits` | `t2wMode` only: the positive woofer command |
| `counter_delta.fits` | `counter` and `both` only: the positive Δ posted |

Every file carries the run header: `DATE-OBS`, `DMSTREAM`, `WFSSHMIM`, `WFSFPS`, `POKEMODE`, `POKEX`/`POKEY` or
`PATFILE`/`PATSHA`, `POKEAMP`, `NDELAYS`, `DLYSPAN`, `NFRAMES`, `NTRIALS`, `NREF`, and the provenance cards:

| Card | Value |
|---|---|
| `EMGAIN` | camWFS `emgain` at run start, `-999` if never received |
| `LOOPST` | `holoop` `loop_state` at run start: `ON`, `OFF`, or `UNKNOWN` if never received |
| `LOOPCHG` | 1 if `loop_state` changed during the run, as of when the file was written; the run is not aborted |
| `GITSHA1` | git SHA-1 of the MagAOX build |
| `GITMOD` | 1 if the build tree had uncommitted changes |

In `t2wMode` the header has `T2WMODE`, `WMFILE` and `WMSHA` instead of `POKEX`/`POKEY` or `PATFILE`/`PATSHA`.  Every file
has `OFFCASE`.  In `counter` and `both` it also has:
- `TMFILE`, `TMSHA`: the tweeter mode cube;
- `CNTSTRM`: the stream written;
- `POSTDLY`: the post delay;
- `CORAMP0`, `CORAMPD`, `COGAIN`, `COLEAK`: counterOffload's settings at the start (`-999` if unknown);
- `COCHG`: 1 if any of those settings changed during the run.

The cubes also have `DELAYUS`, `DLYIDX`, `DLYMEAN`, `DLYSTD`, `DLYMIN`, `DLYMAX`, `NINVALID`, `LATEFRAC`, `LATEFLAG`,
`T50`, `RISE`, `JITTER`, `SETTLED`, `PEAKABS`, and `RESIDRMS`. In `counter` and `both` they also have `POSTMEAN` and
`POSTSTD`.
A header value that can not be computed (e.g. no 10%/90% crossing) is written as `-999`.

Live shared memory, updated during a run: `<name>_ref` (P), `<name>_resp` (mean response curves, (N+1) x K), and
`<name>_respavg` (super-sampled response).

# SAFETY

- Use a DM channel that no other process writes (default `dm00disp07`), with the AO loop open.
- Start with a small `poke_amp`.  `poke.maxCommand` limits the command in both modes.
- The DM channel is zeroed after every trial and whenever the run ends, including on stop, error, and shutdown.
- In `counter` and `both` the app writes `aol0_tweeterOffload`, which belongs to t2wOffloader. It does so only while
  t2wOffloader reports `offload` off, and it requests counterOffload's zero after every trial that posted Δ and at
  the end of the run.
- When t2wOffloader resumes offloading, its next post overwrites the app's last frame.
- The app never writes the tweeter or a synco channel; the counter's tweeter command comes from counterOffload.

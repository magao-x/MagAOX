# Phase 13 — April 15, 2025 telescope telemetry inspection

Workstation inspection: 2026-11-06, branch `parkertjohnson1/accelSim`, starting
revision `79d927f4`. Input is the **real telescope dataset** supplied by the user:
`~/data/phase13_20250415_083000_083500/`. Original data were opened read-only;
all **329 files** have identical before/after SHA-256 hashes. No telescope
connection, application launch, configuration change, DM command or deployment
was performed. No cross-sensor phase/coherence or predictor replay was attempted.

## Verified contents and coarse coverage

Every array agrees with its manifest. There are no missing/extra NPY files,
duplicate manifest entries, changing chunk shapes/dtypes or nonfinite values.
Manifest order, lexical filename order and encoded timestamp order agree.
Metadata identifies the enclosing bundle as
`bundle_2025-04-15_082718_to_2025-04-15_101736`, the requested selection as
08:30–08:35 UTC, and explicitly says **per-frame acquisition timestamps have not
been extracted**. The only other files are the two manifests and `metadata.json`;
there are no timing arrays, counters, receipts or calibration sidecars.

| Stream | Files × shape | Total frames | Encoded first / last file labels, UTC |
| --- | --- | ---: | --- |
| `acc000` | 75 × (8192, 8), UINT16 | 614,400 | 08:29:56.737200135 / 08:34:59.841201682 |
| `aol1_modevalWFS`, reduced to `wfs_tiptilt` | 251 × (2402, 2), FLOAT32 | 602,902 | 08:29:58.895666422 / 08:34:59.145846484 |

Accelerometer filename intervals are 4.095943505–4.096081105 s, median
4.095997016 s. WFS intervals are 1.200896263–1.201099687 s, median
1.201000831 s. Counting all preceding-file rows over the first-to-last label span
gives bulk recorded-frame densities **1999.999990 / 1999.998801 frames/s**.
These are evidence for approximately **2 kHz recorded data**, not a verified
8-kHz source or verified per-frame sampling rate. They cannot distinguish a
2-kHz producer from higher-rate acquisition with skipped/selected records.

At a conditional 2000 recorded frames/s, the row counts represent 307.200 s and
301.451 s. The full chunks include labels before the requested 08:30 start.
Actual last acquisition times, exact cropping, simultaneous acquisition coverage,
lost samples and frame jitter remain unknown. No per-frame or last-frame
acquisition timestamps were reconstructed from filenames. Archive-label interval
variations are not per-frame jitter or cross-host clock-offset measurements.
The manifest's floating epoch labels lose up to about 119 ns; the inspection
parses all nine nanosecond digits directly from names for label bookkeeping only.

## Filename semantics: source evidence versus dataset verification

`apps/streamWriter/streamWriter.hpp::doEncode`, around lines 2088–2124, copies the
saved timing records then takes `m_timingCircBuff + saveStart*5 + 1` for the name.
That is the **first saved image's acquisition field**, not the last image, encode
completion, file creation or first write timestamp. The same assignment and the
UTC `YYYYMMDDHHMMSSNNNNNNNNN` formatting exist in the locally available
pre-observation version `aa86fac3fabf9c60cf703bf15212450a9a790e2a` (2025-03-16),
around lines 1887–1910. This historical source supports the convention but does
not establish which recorder binary actually made these files.

Both versions save five UINT64 words per frame:
`[cnt0, acq_sec, acq_nsec, write_sec, write_nsec]`. Both also replace missing
acquisition seconds with the recorder's realtime clock and missing write time
with acquisition time. Thus even agreement of a filename with the first timing
row cannot alone prove physical acquisition timing. The actual recorder/producer
versions and acquisition-field semantics must be supplied.

The NPY reductions omit that second XRIF timing block. The original archives
are not present on this Mac. **Export their original timing records separately,
or copy the original XRIF files locally before extraction.** An authorized human
must do any work involving files on `exao1`; no agent should run there or connect.
No instrument commands were executed in this phase.

## Signals and spectra supported by current data

Statistics below use the received row order and raw values; no simulator ADC
calibration, physical axis registration, sensitivity or WFS angular scale is
assumed. Mode 0=tip and mode 1=tilt are confirmed by the user/metadata. Their
physical units and optical normalization are not provided.

| Accel channel | Range [counts] | Mean [counts] | Fluctuation RMS [counts] |
| --- | ---: | ---: | ---: |
| 0 | 421–812 | 662.670 | 11.927 |
| 1 | 265–848 | 665.528 | 12.511 |
| 2 | 528–755 | 663.399 | 5.864 |
| 3 | 0–11 | 0.547 | 0.567 |
| 4 | 0–2 | 0.617 | 0.486 |
| 5 | 0–1 | 0.192 | 0.394 |
| 6 | 0–1 | 0.0116 | 0.107 |
| 7 | 0–2 | 0.0271 | 0.162 |

Channels **0–2 are the signal-bearing candidates**. Channels 3–7 have small,
near-zero/few-code values, without comparable target-band peaks. This does not
identify them as digital, unused, broken or physically registered axes. Do not
automatically choose a two-channel pair for pointing. The WFS fluctuation RMS is
**0.0141503 tip / 0.0186174 tilt in recorded modal units**, not radians or mas.

Spectra average complete-file Hann periodograms, subtracting each file's mean.
No window joins an archive boundary. The primary frequency is cycles per received
frame. The table/plots use an explicitly **conditional 2000-frames/s axis**; this
is not a claim of verified sampling or physical calibration. ADC PSD units are
counts²/Hz, WFS PSD units are recorded-modal-unit²/Hz, under that same condition.
Bin spacings are 0.244141 Hz (ADC) and 0.832639 Hz (WFS); bins are not precise
line-frequency estimates. Unknown intra-file loss/decimation can distort them.

| Channel / mode | Conditional peak near 28–29 Hz | Conditional peak near 56 Hz | Peak/background contrast, 28 / 56 |
| --- | ---: | ---: | ---: |
| Accel 0 | 28.564 Hz | 55.908 Hz | 17.34 / 14.71 dB |
| Accel 1 | 28.564 Hz | 55.908 Hz | 5.71 / 14.73 dB |
| Accel 2 | 28.564 Hz | 55.908 Hz | 17.48 / 11.87 dB |
| WFS tip | 28.310 Hz | 55.787 Hz | 19.47 / 6.86 dB |
| WFS tilt | 28.310 Hz | 55.787 Hz | 18.43 / 10.12 dB |

Contrast is the peak-bin power divided by the median nearby sideband power,
excluding the search band plus a 2-Hz guard. It is descriptive, not calibrated
SNR, a statistical detection significance, coherence or a coupling estimate.
Channels 3–7 have only about 0.5–1.4 dB contrast by the same calculation.
Common-looking spectral bands motivate synchronized analysis but do not establish
that the accelerometers predict WFS motion. Different bin grids account for some
peak-bin differences; neither phase nor frequency locking was established.

## Exact timing export required

Preserve one row for **every original frame**, including bad/zero timestamps;
do not interpolate, filter, sort within archives, renumber counters or crop first.
Export separate per-archive timing sidecars and an association manifest containing:

- Exact `source_file`, zero-based source-frame index, original UINT64 `cnt0`,
  acquisition sec/nsec and write sec/nsec. Keep integer components, not float epoch
  seconds. The raw timing array should have shape `(n_frames, 5)` and dtype UINT64.
- Original image and timing header dimensions/dtypes/frame counts, XRIF SHA-256,
  reduced NPY SHA-256, decoder/reduction version or script, and the precise channel/
  mode selection. Confirm no temporal reduction, reordering or filtering occurred.
  Timing row `i` must correspond to reduced array row `i`; otherwise export the
  explicit original-row selection map. Expected frame counts are 8192/2402.
- Recorder/producer app names, hosts, versions/configuration and timestamp meaning,
  reset/restart/stream-identity evidence, loss logs and clock synchronization/step
  evidence. XRIF timing has no original inode/generation/consumer-receipt field.
  Later analysis segment IDs must not be presented as recorded generation IDs.
- ADC channel wiring/physical axes, offset, volts/count, sensor sensitivity/units,
  true acquisition/scan rate and possible channel aperture offsets. WFS exposure
  timing/frame-center convention, mode normalization/optical gain, reference/
  reconstruction and AO state/gains/applied-command context are also required.
  WFS mode-stream acquisition time is not automatically the originating exposure
  center. Clock registration between producer hosts must be independently justified.

Two existing, file-only extraction routes are supported by repository source:

1. **Reuse the original reducer's XRIF reader**, retaining both concatenated
   blocks. `magaox-python/magaox/quicklook/pack.py::unpack_one_xrif` demonstrates
   the `fixr` API below. The second read is timing, not another image archive:

   ```python
   # Run by a human on a workstation/copy of the original archive.
   # fixr is an existing MagAO-X dependency, not installed or tested here.
   with original_xrif.open("rb") as f:
       image = fixr.XrifReader(f).copy_data()
       timing = fixr.XrifReader(f).copy_data().reshape(-1, 5)
   assert timing.dtype == np.dtype("uint64")
   assert image.shape[1] == 1  # verify frame/plane interpretation, do not silently flatten
   assert len(timing) == image.shape[0] == expected_reduced_frames
   np.save(new_output / (original_xrif.stem + "_timing.npy"), timing, allow_pickle=False)
   # Also compare the original ADC 8-channel payload or WFS modes [0,1]
   # with the received reduction, row-for-row, and hash both input and output.
   ```

   Import `fixr`, `numpy as np`, and `Path` in that environment and set the input,
   output and expected count from each existing manifest record. Output must be a
   new directory, outside the original archives and this received dataset. Do not
   reuse the quicklook helper's zero-row filtering; the raw fault rows are needed.

2. **Existing `xrif2fits`, metadata only, one archive per fresh output directory:**

   ```bash
   xrif2fits -N --metaOnly --show-details -f /local/copied-originals/acc000_20250415082956737200135.xrif \
     -D /tmp/phase13-timing/acc000_20250415082956737200135
   xrif2fits -N --metaOnly --show-details -f /local/copied-originals/aol1_modevalWFS_20250415082958895666422.xrif \
     -D /tmp/phase13-timing/aol1_modevalWFS_20250415082958895666422
   ```

   Repeat for **all 326 source filenames in the supplied manifests**, retaining a
   per-file mapping. Each `meta_data.txt` line starts with ISO date, counter,
   acquisition sec/nsec and write sec/nsec. Parse the five integer fields directly,
   add the original source filename and sequential original frame index, and check
   8192/2402 rows. Do not use `-T` (summary only), `--noMeta` or cube mode. Do not
   combine archives into one output directory: the per-archive metadata file may
   be overwritten. `-N` avoids inventing camera/header/exposure metadata; it does
   not establish WFS exposure timing. These flags/code paths were source-reviewed;
   the installed native utility/version and original archives were unavailable here.

Prefer copying the original XRIFs to an offline workstation. If the originals
are only on `exao1`, an authorized human can export these file-only sidecars
there and transfer them to the Mac. This report does not authorize agent access,
process restarts or production changes.

## Next replay gates and reproducibility

First audit every timing row and file boundary for counter gaps/rollback, genuine
acquisition/write ordering, source rate, clock steps, restarts and filename versus
first timing-row agreement. Then register WFS exposure centers and producer clocks,
select/physically identify ADC axes and verify ADC/WFS units. Only after these gates
can the strict [Phase 12 pipeline](RECORDED_TELEMETRY.md) estimate transfer/coherence
or score causal target-time predictions. Passive telemetry still cannot establish
DM correction sign, latency or bandwidth.

The current controller's native input requires a **two-channel pair**, so it cannot
directly ingest this eight-channel stream. A later offline adapter must select an
explicitly justified pair; no production reader change is made here. Do not copy
the simulator's 2048/0.0083799/1.02 or 10-m parameters into a real calibration.
Historical consumer receipts/`accelFFCtrl` queries are not present in XRIF; any
counterfactual replay schedule must be labelled assumed, with native availability
and acquisition-to-query latency left unqualified. Linux/private-native validation
remains separate. Default zero gain, disabled gap policy and live configuration
are unchanged.

Reproduce the present inspection on the Mac with the existing scientific Python:

```bash
MPLCONFIGDIR=/tmp/accelFFCtrl-phase13-mpl python apps/accelFFCtrl/tests/inspect_reduced_telemetry.py \
  "$HOME/data/phase13_20250415_083000_083500" --display-rate-hz 2000 --output /tmp/accelFFCtrl-phase13
python -m unittest discover -s apps/accelFFCtrl/tests -p test_inspect_reduced_telemetry.py -v
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-phase13-build telemetry-tools
ACCELFFCTRL_TELEMETRY_REPLAY=/tmp/accelFFCtrl-phase13-build/accelFFCtrlTelemetry_replay \
  make -C apps/accelFFCtrl/tests validation-test PYTHON=python
python -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

Outputs are `inventory.csv`, `channel_statistics.csv`, `conditional_peaks.csv`,
two sample-domain/conditional-Hz PSD CSVs, three figures, `report.json` and all
329 original file hashes. No output contains fabricated per-frame timestamps.
Four new inspection tests pass. The full Python regression has **36 controller
plus four simulator PSD tests**; numerical/sanitizer regression results are recorded
in the engineering plan. No original assertion or production source was changed.

## Follow-up after the original timing export

The user subsequently supplied both original timing arrays. The new
[timed analysis](PHASE13_TIMED_ANALYSIS.md) supersedes the missing-timing blocker
for recorded-field alignment, verifies all archive starts and reports conditional
coherence/held-out forecasts. This inspection snapshot remains the historical
filename-only result. Physical clock/exposure-center calibration is still pending.

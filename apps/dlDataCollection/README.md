# dlDataCollection

For the short operator procedure, see [Collect with another ND filter](QUICKSTART.md).

Collect four padded PWFS pupils with the exact applied 50x50 tweeter commands.
The app starts idle in READY with no input semaphore wait. Prepare a bank once, select it through INDI, and reuse it
for separate ND collections without editing the config or restarting the app.
The operator still changes `fwtelsim`. The collector automatically checks its
named position, `flipwfsf`, and the camwfs frame rate and E.M. gain before Start
and records them in `run.conf`.

## Prepare commands offline

`prepare_bank.py` and `Generate_random_modevals.ipynb` run on a workstation and
never import a DM interface or send commands. Generation requires NumPy and
HCIPy; legacy conversion requires NumPy. The original instrument notebook is
not modified by this change.

Supply the authoritative binary actuator mask in the existing command memory
order: 2500 values in `.npy`, little-endian float32 `.bin`, or CSV, with 1 for
active and 0 for excluded actuators. Keep the calibrated amplitude range and
command units; this tool does not establish a safe stroke or change units.
Set `PTV_MIN` and `PTV_MAX` to that range in micrometers of DM surface before
running the example from this directory:

```sh
python3 prepare_bank.py generate --output /path/to/new_bank \
    --mask /path/to/recorded_actuator_mask.npy --samples 10000 --sets 15 \
    --seed 42 --units um_surface --ptv-min "$PTV_MIN" --ptv-max "$PTV_MAX"
```

The seed, mask, ranges, dependency versions, and checksums are recorded.
Every sample is generated. Piston is subtracted over the active mask only;
excluded actuators remain exactly zero.

Existing CSV banks need one offline conversion. Both the one-value-per-line
and 2500-values-per-row layouts are accepted:

```sh
python3 prepare_bank.py convert --source /path/to/legacy_bank \
    --output /path/to/new_bank --mask /path/to/recorded_actuator_mask.npy \
    --samples 10000 --sets 15 --units um_surface
```

Nonzero excluded actuators cause rejection. `--repair-mask` explicitly zeros
those positions in a new bank and records the transformation; it does not
modify the originals. Existing output directories are always rejected.
An incomplete bank has no valid manifest and cannot start acquisition.

## Collect

Old config keys remain: `parameters.ampsDir` is now a prepared bank directory,
and `parameters.dataDirs` is the parent of unique run directories. `Nperset`
must match the bank; `Nset` selects exactly that many sets. Zero uses the bank
value for either count. `Nmodes` must be 2500; the formerly unused `modalNorm`
must remain 1. `frameTimeout` defaults to one second without a fresh frame.

With the normal instrument prerequisites satisfied and an exclusive, initially
zero `dm01dispXX` channel, the operator can:

1. Set `dlDataCollection.bank.target` and `dlDataCollection.output.target`
   while idle. The configured paths are initial values, not per-run requirements.
2. Request `dlDataCollection.start.request=On`.
3. Read `dlDataCollection.collection`: phase, actual directory, accepted images,
   completed sets, duplicates, torn reads, effective samples/second, and error.
4. Wait for `complete` before changing ND, or request `stop.request=On` and
   wait for `cancelled`. A failure is reported as `failed` with an error.
5. Verify the new ND position, choose another output parent if wanted, and Start.

Start first reports `checking_observation` while requesting fresh
`fwtelsim.filterName`, `flipwfsf.presetName`, both `fsm` properties, and
`camwfs.fps.current` / `camwfs.emgain.current`. The named
positions must be unambiguous and Idle/Ok; controller status must be READY,
OPERATING, or POWEROFF with a retained known position. No filter or flipper motion
is commanded. Missing, moving, disconnected or stale readbacks block Start after
five seconds. Readbacks refresh once per second outside the image callback.
During collection, reported position changes/motion or loss of valid readback
abort the run through the existing partial-data handling. Camera readbacks must
be finite, positive and Idle/Ok. A reported frame-rate or E.M.-gain change, or
loss of valid camera readback, also aborts the run and preserves partial data;
finish or Stop before changing these camera settings.
At each Start the dedicated command channel must be stable and zero. An external
zero while idle is accepted by refreshing the initial counter; changes during
preparation, collection or between sets still abort. Start never zeros the channel.

`observation.nd`, `observation.flipwfsf`, `observation.camwfs_fps` and
`observation.camwfs_emgain` show live readbacks in cursesINDI.
Each run freezes `nd`, `flipwfsf`, `camwfs_fps` (Hz), `camwfs_emgain`
(dimensionless gain factor), controller states, property identifiers and
local Unix readback receipt times in `run.conf`. These are controller reports,
not independent physical measurements. Folder and bank names do not set ND.
`camwfs_fps` is the camera-reported frame rate, distinct from the collector's
`samples_per_second`. Existing training manifests retain these values in each
run's `settings` dictionary; no duplicate image data or new dataset format is needed.

`NumFrameSkip=8` requires a source-counter advance of at least eight after each
command publication. Zero still waits for a fresh frame. Duplicate notifications
and torn snapshots never advance the sample. This records software publication
and source identity; physical DM settling still needs instrument validation.

## Output and interruptions

Use the directory reported in `collection.directory`; old data readers must
point there instead of the parent. Complete sets retain `images_dataset_N.bin`
in the previous float32 four-pupil, column-major layout, with two zero pixels
at each pupil edge. Read one set as:

```python
images = np.fromfile(run / "images_dataset_0.bin", dtype="<f4")
images = images.reshape(samples, 4, pupil, pupil).transpose(0, 1, 3, 2)
commands = np.fromfile(run / "commands_dataset_0.bin", dtype="<f4").reshape(samples, 2500)
```

Each set also has `frames_dataset_N.csv` and `complete_dataset_N.conf`.
Only the completion marker certifies checked, flushed payloads and their hashes.
`run.conf` records bank identity, crop, scaling, source/channel, and revision;
`finished.conf` records the final status. Producer timestamps and host command
timestamps have different clock domains and must not be subtracted blindly.

Stop, source timeout/reset/reconnection, or a lost command channel abort capture
and save the accepted prefix with `partial_dataset_N` names. Slow output pauses
at set boundaries with the owned channel zeroed. Two buffers bound memory
(about 1.51 GB at 10000 samples and four 64x64 pupils). Failed writes retain
buffers and `.part` files, block another Start, and require operator diagnosis
before restart. Never treat unmarked payloads as complete training sets.

## Notebook defaults and automatic training labels

After transferring the updated source, the operator can update their existing
bank notebook once, preserving its layer recipe, calibration, custom raw-data
path and a backup:

```sh
python3 /home/rlandman/MagAOX/apps/dlDataCollection/update_bank_notebook.py \
    --dated-paths /home/rlandman/NNReconstructor_2026B/Generate_random_modevals.ipynb
```

The parameter cell then exposes `BANK_ROOT`, `BANK_DATE` (the notebook computer's
local date in YYYYMMDD), `BANK_NAME`, and the final `BANK_DIR`. Override
`BANK_DATE` for another observing date, change `BANK_NAME` for a new recipe/run,
or set `BANK_DIR` to a full path. Existing banks are neither moved nor overwritten.
The raw-image output directory remains an independent operator setting.

`training_metadata.patch` updates the existing home-folder `nn_data.py` so
`package_run` reads ND from `run.conf` automatically, rejects a conflicting manual
ND label, and allows `Dataset.select(values={"flipwfsf": ["out"]})`. Flipper
positions stay in the existing run metadata; binary array formats are unchanged.
For an older run without automatic metadata, retain its manual ND label; its
flipper position is `unknown`.

Apply the patch as the owner of the notebook folder, after checking it applies
to that version. This does not execute a notebook or alter existing datasets:

```sh
cd /home/rlandman/NNReconstructor_2026B
patch --dry-run --forward --fuzz=0 -p1 < /home/rlandman/MagAOX/apps/dlDataCollection/training_metadata.patch
patch --forward --backup --fuzz=0 -p1 < /home/rlandman/MagAOX/apps/dlDataCollection/training_metadata.patch
```

In `Save_training_data.ipynb`, a `COLLECTIONS` entry can now use `None` as its first
item to read the actual ND automatically. Keep specifying the bank and raw-data
parent. Existing explicit ND entries still work if they match the recorded wheel.
Restart the notebook kernel after updating `nn_data.py`.

The existing training selection can include both recorded fields, for example:

```python
selected = data.select(values={"nd": ["ND2"], "flipwfsf": ["out"]})
```

## Verification

On a workstation with the normal MagAOX development dependencies:

```sh
make -C apps/dlDataCollection
make -C tests -f Makefile.one t=../apps/dlDataCollection/tests/dlDataCollection_test
apps/dlDataCollection/tests/dlDataCollection_test
python3 -m unittest discover -s apps/dlDataCollection/tests -p 'test_*.py'
```

To exercise the loader patch, set `NN_DATA_PATH` to a patched offline `nn_data.py`
when running those Python tests. Without it, only the two optional loader checks
are skipped.

The CPU tests cover capture and actual file handling, with injected hardware,
slow writers, and write failures. The observing-state update passes these tests
with address/undefined sanitizers. The adapter is also smoke-tested with stubbed
base classes and hardware; this is not a full application build. The app disables fast-math so
NaN/Inf validation remains effective. The full Linux app build, live stream
integration, latency, and framerate remain unverified; no deployment or hardware
operation was performed. ND sequencing and camera-setting changes are outside
this patch.

## Input attachment lifecycle

Start requests OPERATING and allows five seconds for source/channel attachment and fresh
observing readbacks before starting the existing session preflight. Stop cancels a pending
Start. Draining and terminal sessions return to READY and release the input mapping/reader
slot. A new Start waits for detachment; an unexpected reconnect during an active session
still aborts the partial run. A successful READY request is asynchronous and can take the
monitor's existing timeout to settle.

CPU session/observation regressions and the shared monitor-loop fixture pass locally.
Real ImageStreamIO producer and collector app-level acceptance remain required on an
isolated Linux host; see the [acceptance plan](../../agents/plans/2026-11/shmimMonitor-idle-apps.md).

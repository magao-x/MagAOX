# PO4AO deployment: existing notebook, then managed training

## Update an existing installation: parameters and saved models

This update requires **po4aoTrainer on exao2 and po4aoGUI in the IOC container**.
Run the commands yourself. Keep your existing configurations and Python environment;
no dependency, native PO4AO, nnReconstructor, or INDI-server update is required.
Pause learning and let any pending model load finish before restarting the trainer.
Restarting loses unsaved in-memory training state; acknowledged publications remain
on disk and can be saved with the new GUI after the update.

Transfer `nn-po4ao-gui-policy-library.bundle` into `/tmp` on both machines, with
read permission for the account running Git. It updates an existing deployment of
this feature branch; preserve local changes if `git status` is not clean.

**On exao2, from your checkout:**

```sh
cd /home/rlandman/MagAOX
git status --short --branch
git fetch /tmp/nn-po4ao-gui-policy-library.bundle rlandman/nn-po4ao-gui
git merge --ff-only FETCH_HEAD
xctrl shutdown po4aoTrainer
TRAIN_PY=/opt/conda/envs/po4aoTrainer/bin/python
make -C apps/po4aoTrainer INSTRUMENT_CONDA_ENV=po4aoTrainer PYTHON="$TRAIN_PY -I" install
sudo install -d -o xsup -g magaox -m 2775 /data/po4aoTrainer/saved_engines
git rev-parse HEAD
```

Set `source_revision` in `/opt/MagAOX/config/po4aoTrainer.conf` to
`rlandman/nn-po4ao-gui@` followed by that commit, then run `xctrl startup po4aoTrainer`.
The library defaults to `<artifact_root>/saved_engines`; only add `saved_engines`
to the app config if using a different absolute path. Provision that path instead
if your artifact root differs from `/data/po4aoTrainer`.

**Inside the working IOC GUI container:**

```sh
cd "$HOME/MagAOX-po4ao-gui"
git status --short --branch
git fetch /tmp/nn-po4ao-gui-policy-library.bundle rlandman/nn-po4ao-gui
git merge --ff-only FETCH_HEAD
make -C gui/apps/po4aoGUI && make -C gui/apps/po4aoGUI install
po4aoGUI po4ao po4aoTrainer nnReconstructor &
```

Close the old GUI window and use the newly launched one. The three fields show
values from your installed training config. Enter submits an edit; the displayed
number remains the acknowledged value, with requested value in its tooltip and
pending state in the status line. Paused edits apply on resume. Edits survive Reset
and Set policy, but a trainer restart restores file settings. A valid file edit to
one of these three keys replaces that key's session override.

**Test the saved-model workflow:**

1. Save an acknowledged trained pair as, for example, `internal-bright`. Save is
   also available for the retained managed pair after Disable or trainer restart.
   If no complete acknowledged pair exists yet, first Reset and train one.
2. Choose it in the Policy dropdown and click **Set policy**. Wait for completion;
   both models are restored in a new paused session. Controller enable state stays
   unchanged, while optimizers/replay/iteration start fresh. Current parameter
   settings remain in force. The engine is rebuilt, so this step can take time and
   temporarily needs memory for both the old and prepared training sessions.
3. Enable PO4AO if needed, click **Recollect warmup**, wait for Ready, then **Resume
   learning**. Recollection applies exploration noise to the loaded policy. Set
   policy alone does not inject noise or start learning.
4. Save again after a newly trained engine is acknowledged. Reusing a label creates
   another timestamped entry; Reset never removes saved entries. The delta-policy
   penalty follows current acknowledged runtime policy changes, including each
   newly activated engine, rather than staying tied to the initialization.

Matching dimensions does not establish modal-basis/calibration compatibility;
verify the instrument setup yourself. Loop state and routing remain operator-owned.
GPU export and actual control improvement still require instrument acceptance.
If an interrupted load shows **Set policy or Reset required**, resolve it with one
of those actions before resuming. Before on-sky use, push the tested source and
configuration revisions to the approved remotes and record their identities, as
required by the [development policy](https://magao-x.org/docs/handbook/software/hacking.html).

**Optional progress notebook**, on exao2:

```sh
cp -i notebooks/po4ao/watch_training.ipynb /home/rlandman/PO4AO_2026/po4ao_training_progress.ipynb
"$TRAIN_PY" -m ipykernel install --user --name po4aoTrainer --display-name "PO4AO trainer"
```

Open `po4ao_training_progress.ipynb` in your existing Jupyter interface and select
**PO4AO trainer**. Run the notebook cells; set `AUTO_REFRESH = True`
in its final cell to watch continuously, or rerun `latest_record = refresh()`.
Interrupting this observer notebook leaves training running. It reads files only;
keep the original training notebook's training/export cells stopped.

Plots normally update every 50 iterations after any publication finishes. The
notebook shows run identity and measurement age, residual RMS/PSD, model diagnostics,
publication outcomes and timing fields. While an export is running, the publication
record may be newer than the plots. Pausing training also pauses snapshot updates.
Use live instrument measurements for paused-policy PO4AO/integrator comparisons;
predicted cost reductions alone do not establish closed-loop improvement.

The remaining sections cover a first installation or a full native-app deployment.

Use the same `po4aoGUI` for NN selection, notebook-based PO4AO, and optional managed
training. **NN controls require only `nnReconstructor`; no trainer installation is
needed.** You run the commands and manage loop state. All checkouts must contain the
latest `rlandman/nn-po4ao-gui` changes (the older `2a6f180e` bundle predates this fix).
The exao2 checkout is `/home/rlandman/MagAOX`; the GUI runs inside your MagAO-X
container on `ioc.strw.leidenuniv.nl`.

## 1. Build/install the native apps on exao2

As `rlandman`, save the installed binary/configuration and build the two native apps before stopping either one:

```sh
cd /home/rlandman/MagAOX
git status --short --branch
# Update a clean checkout from the new readable bundle copied here:
git fetch /tmp/nn-po4ao-gui.bundle rlandman/nn-po4ao-gui && git merge --ff-only FETCH_HEAD
BACKUP="$HOME/po4ao-backup-$(date +%Y%m%d-%H%M%S)"
mkdir "$BACKUP"
cp /opt/MagAOX/bin/{po4ao,nnReconstructor} "$BACKUP/"
cp -L /opt/MagAOX/config/{po4ao.conf,nnReconstructor.conf,isRTC.conf,proclist_RTC.txt} "$BACKUP/"
readlink -f /milk/shm/aol1_mfiltINPUT.im.shm > "$BACKUP/mfilt-input.txt"
cat /milk/shm/aol1_calib_source.txt > "$BACKUP/calibration.txt"
make -C apps/nnReconstructor clean && make -C apps/nnReconstructor && \
  make -C apps/po4ao clean && make -C apps/po4ao
# Only after both builds succeed:
xctrl shutdown nnReconstructor && make -C apps/nnReconstructor install && \
  xctrl startup nnReconstructor
xctrl shutdown po4ao && make -C apps/po4ao install && xctrl startup po4ao
xctrl status nnReconstructor
sudo -u xsup tmux capture-pane -p -t nnReconstructor -S -80
xctrl status po4ao
sudo -u xsup tmux capture-pane -p -t po4ao -S -80
```

Keep the backup path and verify the recorded route/calibration identify the previous
setup; record them manually if either file is unavailable. Keep `po4ao.conf`
unchanged for notebook testing. CUDA/TensorRT dependencies must build successfully
before stopping an old app. The NN build uses the already installed milk headers
under `/usr/local/milk/include` for FPS readback; it needs no Python milk module.
No dlDataCollection or INDI-server reinstall is needed. If the PO4AO changes from
this branch are already installed, skip its rebuild/restart; this correction changes
only NN, the GUI, and the optional trainer.

## 2. Build/install the GUI inside the IOC container

Use the same container in which the other MagAO-X GUIs and `qmake` work. Transfer
the new Git bundle into it using the method that worked previously, make it readable
by your container user, and update the existing GUI checkout. Substitute its actual
path below; use a plain path, not a `file://` URL:

```sh
cd "$HOME/MagAOX-po4ao-gui"
git status --short --branch
# With a clean checkout (preserve any local edits first):
git fetch /tmp/nn-po4ao-gui.bundle rlandman/nn-po4ao-gui
git merge --ff-only FETCH_HEAD
make -C gui/apps/po4aoGUI && make -C gui/apps/po4aoGUI install
po4aoGUI po4ao po4aoTrainer nnReconstructor &
```

Run `make install` as the normal build user; the install rule itself uses sudo.
For a fresh checkout only, first build `INDI/liblilxml` and `INDI/libcommon` with
`make -C`. No Qt installation on the outer IOC host or your laptop is needed.
The container's existing localhost:7624 endpoint must expose `nnReconstructor` and
`po4ao`. In the operator's working deployment these arrive through **isAOC on
exao1**. Its imported RTC device list must also contain `po4aoTrainer` for managed
training. After changing that list on exao1, run `xctrl restart isAOC` there; this
was the missing step when the trainer heartbeat worked on exao2 but not at the GUI.
A container using only a tunnel needs no local driver-list edit or server restart.

With the trainer stopped, verify the NN catalog appears, choose a model and use
**Set reconstructor**. The adjacent button should read **Switch to CNN** when CACAO
is observed and **Switch to CACAO** when CNN is observed. Check the loaded/active
model readbacks and actual writer after each completed request. Link modulation is
optional (CNN unmodulated; CACAO 2 kHz/radius 3). These controls never change loop
state or downstream mfilt routing. An old NN binary can populate the catalog but
cannot provide the new control properties: update NN as well as the GUI.

## 3. Test with the existing notebook

Leave `po4aoTrainer` stopped; no trainer installation is needed. Open
`/home/rlandman/PO4AO_2026/po4ao_notebook_nonlinear.ipynb` using its existing **torchpy**
kernel. Keep its training/export cells and plots; reopen stream handles after the
runtime restart. The GUI should show **PO4AO ready / trainer unavailable**.

In the HOWFS loop root directory on exao2, perform the downstream setup manually:

```sh
cd /opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-rootdir
howfs_mfilt_switch po4ao_output && cacao-calib-apply default
readlink -f /milk/shm/aol1_mfiltINPUT.im.shm
```

The installed scripts restart CACAO processes and apply calibration. After both
commands, verify mfilt actually consumes `po4ao_output` and its counters advance;
recheck the intended CACAO/NN writer. A symlink or PO4AO model ID alone does not prove
consumption. Manage the loop yourself; the GUI never runs these setup commands.

| Test | Expected result |
| --- | --- |
| Notebook export/reload, then Enable → Disable → Enable | Same loaded policy; effective controller switches policy/integrator/policy. |
| Recollect warmup | Exploration on the selected controller; collection reaches its target and retains that controller. Refresh notebook replay yourself. |
| Reset, including while Enable awaits frames | Off and Reset are sent immediately; completion waits for runtime frames. No learned policy remains. Reset notebook models yourself if intended. |
| Start/Pause learning | Disabled. Training remains controlled by the notebook. |

Pause notebook iteration/export around Reset or replay replacement. Closing the
GUI changes neither training nor inference. NN controls remain independently available.

## 4. Add managed training later, on exao2

Stop notebook training/export. The managed trainer starts its own models/optimizer;
it does not adopt the notebook's session. Clone the known notebook environment into
a shared **new** environment; leave the original untouched. The trainer no longer
uses `CacaoProcessTools`, so no milk Python module build is needed:

```sh
cd /home/rlandman/MagAOX
sudo /opt/conda/bin/conda create -y --copy -n po4aoTrainer --clone /opt/conda/envs/torchpy
TRAIN_PY=/opt/conda/envs/po4aoTrainer/bin/python
sudo -H "$TRAIN_PY" -I -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
sudo -H "$TRAIN_PY" -I -m pip install \
  numpy==2.0.0 onnx==1.20.1 onnxscript==0.6.2 onnx-ir==0.2.0 \
  'xconf @ git+https://github.com/xwcl/xconf.git@d6f4de40bd290510bfe191b777d85d36b467d0ce' \
  'purepyindi2 @ git+https://github.com/xwcl/purepyindi2.git@6595f5770d84fcf010223e317e07c970a8ba3f17' \
  ./magaox-python
make -C apps/po4aoTrainer INSTRUMENT_CONDA_ENV=po4aoTrainer PYTHON="$TRAIN_PY -I" install
cp -i apps/po4aoTrainer/configs/{po4aoTrainer.conf,po4ao_training.conf} /opt/MagAOX/config/
sudo install -d -o xsup -g magaox -m 2775 \
  /data/po4aoTrainer /data/po4aoTrainer/saved_engines \
  /opt/MagAOX/logs/po4aoTrainer /opt/MagAOX/telem/po4aoTrainer
"$TRAIN_PY" -I -m pip check
```

Skip environment creation if it already exists. Cloning does not copy the notebook's
user-site packages; the explicit numerical versions above match the previously
inspected notebook environment. Isolated Python (`-I`) avoids imports from a home
directory. For an older trainer config, remove
`nn_device`, `modulation_device`, `fps_name`, `fps_fifo`, `writer_process_match`,
`unmodulated_state`, and `modulated_state`; NN configuration now belongs in
`nnReconstructor.conf` (`recon` section). Review the two config files:

- `po4aoTrainer.conf`: actual PO4AO device name, `artifact_root="/data/po4aoTrainer"`,
  and `source_revision="rlandman/nn-po4ao-gui@<git rev-parse HEAD>"`. The configured
  TensorRT builder exists at `/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec`.
  Keep `sleep_interval_sec = 2.0`: the driver needs a one-second quiet gap during
  startup. The older 0.2-second setting can keep it flushing output indefinitely.
- `po4ao_training.conf`: training settings and GPU assignments. Current defaults use
  GPU 1 for training/building, with runtime inference on GPU 0; verify compatibility.
  The notebook's `ACTION_PENALTY_WEIGHT` is `[RL] loss_function_penalty = 0.01`;
  `POLICY_UPDATE_PENALTY_WEIGHT` is `[policy_update] penalty = 0.5`. Older examples
  used 0.2 and 1.0 respectively; edit the existing keys to use the operator's values.
  Both support hot updates at a training iteration boundary (on resume if paused),
  without restarting the app. Keep the other keys in those sections.
  Install the updated trainer before using `penalty`; older installed versions
  require `weight`. The updated app accepts either spelling, but rejects both in
  one section. Renaming the key alone preserves the resolved config revision.
- Existing `po4ao.conf`: keep geometry/gains/limits. Its current `10 * 1000 == 10000`
  warmup/replay dimensions already satisfy the trainer. Change only the engine path:

```ini
[parameters]
engineDirs=/data/po4aoTrainer/current
engineName=policy.trt
```

The distributed training example now matches all 33 explicit settings in the
operator's 2026-11-03 nonlinear notebook excerpt, except the two penalty values
deliberately kept at 0.01 and 0.5. The excerpt's later FAINT assignments would
otherwise override them to 0.1 and 1.0. Six other defaults changed:

| Training setting | Earlier example | Notebook-matched example |
| --- | --- | --- |
| `NN_models.training_batch` | 256 | 128 |
| `nonlinear.channels` | 8 | 32 |
| `nonlinear.shared_layers` | 2 | 3 |
| `training.model_publish_interval` | 100 | 50 |
| `training.first_model_publish_iteration` | 250 | 150 |
| `training.max_candidate_action_rms_ratio` | 1.5 | 5.0 |

Changing network channels/layers requires Reset before the worker adopts the new
architecture. The excerpt inherits geometry, stream paths, command limits, gains
and compute settings from `po4ao_config`, which was not supplied; parity of those
inherited settings is not established. Diagnostic sample rate also remains the
existing 2000 Hz setting. Existing installed files are not overwritten automatically.

Publication creates `current`; do not make it a directory. Add
`po4aoTrainer po4aoTrainer` to `proclist_RTC.txt` and append `po4aoTrainer` to
`isRTC.conf`'s `[local] drivers` on exao2. Expose it through exao1's isAOC as in
step 2. For this initial registration, restart the affected servers: `xctrl restart
isRTC` on exao2 and `xctrl restart isAOC` on exao1 after their respective list changes.
Skip server restarts when updating an already connected trainer.

Run the checker as the runtime account (`xsup`); then start the apps:

```sh
cd /tmp
sudo -u xsup /opt/conda/envs/po4aoTrainer/bin/python -I -m xapp.po4aoTrainer.check_deployment
xctrl shutdown po4ao && xctrl startup po4ao
xctrl startup po4aoTrainer
xctrl status po4aoTrainer
sudo -u xsup tmux capture-pane -p -t po4aoTrainer -S -80
```

In the same GUI: **Reset → wait → Enable PO4AO → collection Ready/full target →
Start learning**. Start alone does not acquire initial warmup. Reverify downstream
routing after configuration changes. Confirm real export/reload and matching model ID,
then test Pause/Resume, Disable/Enable, Recollect and Reset. Disable pauses learning;
Enable restores the retained policy without resuming training. Recollect preserves
the controller and resumes only previously running learning. Reset stops the worker.

Valid hot config edits apply during training; reset-only changes are shown in the
GUI. Before accepting managed operation, test delayed loads, other-client operations,
trainer loss/return and status recovery, and measure latency during training/export.

## Recovery

If the trainer and driver run but no heartbeat appears on exao2, check:

```sh
logdump -n 1 isRTC 2>&1 | grep 'po4aoTrainer' | tail -30
```

A final `Starting flush ... po4aoTrainer.out` without completion indicates that
continuous trainer updates prevent the driver's one-second quiet gap. Change
`sleep_interval_sec = 0.2` to `sleep_interval_sec = 2.0` in
`/opt/MagAOX/config/po4aoTrainer.conf`, then restart only the trainer:

```sh
xctrl shutdown po4aoTrainer && xctrl startup po4aoTrainer
```

Allow the driver to reconnect, then run this twice a few seconds apart:

```sh
getINDI -t 3 'po4aoTrainer.status.heartbeat'
```

The counter must increase. This changes the supervisor/status interval, not the
C++ inference or numerical training-loop rate; no package reinstall or isRTC
restart is required for this correction. Check the same heartbeat on exao1 and
in the GUI container only after it works on exao2. Remote-server configurations
can import local copies of `isRTC.conf` through `[remote] servers`; update the
configuration actually used by each server instead of assuming every hop needs
an explicit `[remote] drivers` entry. A direct TCP tunnel needs no device list.

`Session po4aoTrainer does not exist` is normal on first startup; it should be
followed by session creation. Use `xctrl status po4aoTrainer` and the tmux capture
above to distinguish a missing session from a crashed app. In this `xctrl` version,
`peek` calls the missing `_brief_status` function; its `NameError` is a log-command
failure, not evidence that the app failed.

If an earlier installation used purepyindi2 `c26a8276`, its FIFO constructor fails
with `os.open(None)` when `indiserver.ctrl` exists. Replace only that package in the
dedicated environment, then start the failed trainer:

```sh
sudo -H /opt/conda/envs/po4aoTrainer/bin/python -I -m pip install --no-deps --force-reinstall \
  'purepyindi2 @ git+https://github.com/xwcl/purepyindi2.git@6595f5770d84fcf010223e317e07c970a8ba3f17'
xctrl startup po4aoTrainer
xctrl status po4aoTrainer
sudo -u xsup tmux capture-pane -p -t po4aoTrainer -S -80
```

This revision keeps the standard configured-driver FIFO handshake and predates the
broken automatic registration code. No environment rebuild or INDI-server restart
is needed for this dependency correction. Use `startup` for this failed process:
in this `xctrl` version, `restart po4aoTrainer` also adds the role's INDI server.

On Alert, inspect readback/logs before retrying. Missing frames can leave Reset
pending; a lost managed worker session requires Reset. To return to the notebook,
stop managed training/export and `xctrl shutdown po4aoTrainer`, restore the saved
notebook engine path, and restart `po4ao`.

For full rollback, stop the affected apps, restore the saved configs and install the
saved binaries with `sudo install -m 4755 -o root "$BACKUP/po4ao" "$BACKUP/nnReconstructor" /opt/MagAOX/bin/`;
restart only affected processes/INDI servers. Manually restore the **recorded** mfilt
input and calibration, then verify consumption again. Restore the old GUI if replaced.
Stopping the trainer alone does not stop C++ inference.

Target builds, permissions, GPU export/reload, live INDI/FPS, routing and timing remain
operator acceptance checks. Commit/push software and configuration before scientific
use, following the [MagAO-X development guide](https://magao-x.org/docs/handbook/software/hacking.html).

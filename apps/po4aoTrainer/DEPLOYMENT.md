# PO4AO GUI deployment: notebook first, managed training later

Use **one GUI for both workflows**: `po4aoGUI po4ao po4aoTrainer`. The trainer is
optional. With it stopped, Enable/Disable, Reset and Recollect operate directly on
the C++ `po4ao` app; Start/Pause is greyed out. Your existing live notebook keeps its
training cells, environment and plots. The supplied synthetic notebook is optional.

These are human-operator steps for the 2026-11-03 trial. No instrument deployment
or GPU/live-INDI validation has been performed by the agent. You manage loop state.
Examples use the device names `po4ao` and `po4aoTrainer`; substitute actual names.

## 1. Install the runtime and GUI

Use the normal deployment account/build environment (normally `xdev`). Make branch
`rlandman/nn-po4ao-gui` available on both build hosts through your normal Git workflow;
The reviewed fixes require commit `be3f7652` or later; local agent commits have
**not been pushed**. Preserve the previous installed source,
config revisions and binaries for rollback. Record the current downstream mfilt
input stream and active calibration/setup as well; do not infer them from PO4AO
status. On each host, set the checkout path:

```sh
MAGAOX_SRC=/opt/MagAOX/source/MagAOX
git -C "$MAGAOX_SRC" status --short --branch
git -C "$MAGAOX_SRC" rev-parse HEAD
```

Confirm the intended branch and clean deployment checkout. On **RTC**, build/install
only the updated runtime, arranging the process transition before installation:

```sh
make -C "$MAGAOX_SRC/apps/po4ao"
xctrl shutdown po4ao
make -C "$MAGAOX_SRC/apps/po4ao" install
```

This requires the target CUDA/TensorRT SDKs; the app Makefile uses `/usr/local/cuda`
and `/opt/MagAOX/vendor/TensorRT-10.0.0.6`. Keep the existing `po4ao.conf` geometry,
gains, limits, stream names and notebook engine path (`parameters.engineDirs` plus
`engineName`). C++ starts without an active learned policy; the notebook must export
and request `reload_engine` as before. Check its export dimensions/path match this
runtime. Reopen notebook stream handles if the restart recreated streams.

Ensure `proclist_RTC.txt` contains `po4ao po4ao`, and append `po4ao` to the existing
`[local] drivers` list in `isRTC.conf` if absent. Do not replace other entries.
The GUI host's normal localhost:7624 aggregation must expose this device. If it uses
an explicit `[remote] drivers` list, append `po4ao@<existing-RTC-tunnel-name>`;
whole-server aggregation may already include it. Apply changed server configuration
with named `xctrl restart isRTC` on RTC and a named restart of the affected aggregator
on the GUI host, only if edited. This briefly interrupts those INDI routes.

```sh
xctrl startup po4ao
xctrl status po4ao
xctrl peek po4ao
```

On the **GUI host**, from the same revision:

```sh
make -C "$MAGAOX_SRC/gui/apps/po4aoGUI"
make -C "$MAGAOX_SRC/gui/apps/po4aoGUI" install
po4aoGUI po4ao po4aoTrainer
```

The installed executable is `/usr/local/bin/po4aoGUI`; it replaces the previous
`learningCtrlGUI` launcher. No new startup window, tunnel or service is installed.
No NN or `dlDataCollection` rebuild is required for this notebook/PO4AO test.

## 2. Test with your existing notebook

If the trainer is already registered/running, stop it with
`xctrl shutdown po4aoTrainer` on RTC. Otherwise skip this step; **do not install or
start it for notebook testing**. Use the existing notebook, e.g.
`/home/rlandman/PO4AO_2026/po4ao_notebook_nonlinear.ipynb`, with its working kernel on
RTC and normal Jupyter access. No migration to the packaged training core is required.

The GUI should show **PO4AO ready / trainer unavailable** once the runtime heartbeat
advances. If it stays unavailable, inspect `po4ao.fsm`, `model_status` and switch
properties in the normal INDI client; this runtime must publish `instance` and
advancing `sequence`. Merely seeing a process or an old model ID is insufficient.

| GUI action | Notebook workflow |
| --- | --- |
| Enable / Disable PO4AO | Select the resident learned policy / integrator through `control_enable.toggle`. Disable retains the loaded engine. Export/reload remains notebook code. |
| Reset PO4AO | Request control Off, then `reset_model.toggle=On`. This also cancels an Enable awaiting frames; completion uses native request/applied generations. Reset notebook Python models yourself if intended. |
| Recollect warmup | Request `recollect_warmup.toggle=On`; collect with exploration on the current controller and keep that controller selected. |
| Start / Pause learning | Unavailable. Start/pause training and manage replay refresh in the notebook. |

Before comparing correction, select the downstream route **manually** on RTC in
the normal CACAO environment:

```sh
howfs_mfilt_switch po4ao_output && cacao-calib-apply default
```

After **both** commands, verify in the live CACAO/mfilt configuration that its input
is `po4ao_output`, matching the configured PO4AO output, and verify its input/frame
counters advance as that stream updates. Record that evidence. If the route cannot
be verified, stop the comparison. The GUI/supervisor does not run these commands.
PO4AO model/controller telemetry describes emitted output; it does not establish
that CACAO consumes it. Actual instrument routing remains unverified here.

Test Enable → Disable → Enable after a notebook export/reload: the loaded ID should
remain the same while the effective controller changes. Test Recollect on that
policy: progress reaches its target and the controller stays selected. Test Reset:
the runtime returns to integrator with no active learned policy. A request remains
Busy until the required runtime frames complete it. Pause notebook iteration/export
when necessary to prevent your own code from republishing during Reset or using old
replay during recollection. Closing the GUI changes neither training nor inference.
The retained NN controls need the companion and are unavailable in this stage.
Also test Reset after Enable is waiting for input frames: Reset must remain
available and complete when frames return, without a stranded Busy indication.

## 3. Add managed Python training later

Keep the same GUI open. Stop notebook training/export yourself. The managed app
starts its own session; it does not adopt notebook models or optimizer state.

**Environment/install, on RTC:** create a shared Python 3.11+ environment by cloning
the known working notebook environment, retaining CUDA/torch, ImageStreamIOWrap and
CacaoProcessTools. Run environment creation only if it does not already exist:

```sh
NOTEBOOK_ENV=/absolute/path/to/working/conda/environment
sudo /opt/conda/bin/conda create -y -n po4aoTrainer --clone "$NOTEBOOK_ENV"
TRAIN_PY=/opt/conda/envs/po4aoTrainer/bin/python
sudo -H "$TRAIN_PY" -m pip install \
  'xconf @ git+https://github.com/xwcl/xconf.git@d6f4de40bd290510bfe191b777d85d36b467d0ce' \
  'purepyindi2 @ git+https://github.com/xwcl/purepyindi2.git@c26a8276c95b95aef145c50439e291b782eab190'
sudo -H "$TRAIN_PY" -m pip install "$MAGAOX_SRC/magaox-python"
make -C "$MAGAOX_SRC/apps/po4aoTrainer" INSTRUMENT_CONDA_ENV=po4aoTrainer -n install
make -C "$MAGAOX_SRC/apps/po4aoTrainer" INSTRUMENT_CONDA_ENV=po4aoTrainer install
```

The pinned framework revisions above passed local tests; verify their target-stack
compatibility. Check the dry run selects `TRAIN_PY`. Installation links
`/opt/MagAOX/bin/po4aoTrainer`; do not change the general instrument environment.

**Configuration/storage:** copy the two supplied templates into the deployed config
repository if new; preserve/edit existing versions rather than overwriting them:

```sh
cp -i "$MAGAOX_SRC/apps/po4aoTrainer/configs/po4aoTrainer.conf" /opt/MagAOX/config/
cp -i "$MAGAOX_SRC/apps/po4aoTrainer/configs/po4ao_training.conf" /opt/MagAOX/config/
sudo install -d -o xsup -g "$(id -gn xsup)" -m 2755 \
  /data/po4aoTrainer /opt/MagAOX/logs/po4aoTrainer /opt/MagAOX/telem/po4aoTrainer
```

Use the actual runtime account if different from `xsup`. In `po4aoTrainer.conf`, set
actual NN/PO4AO/modulation device names, FPS/FIFO/writer identity, config paths,
`artifact_root="/data/po4aoTrainer"`, and `source_revision="rlandman/nn-po4ao-gui@<full-commit>"`.
The default `trtexec` is `/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec`;
override it in the app config if the installed builder differs.
In `po4ao_training.conf`, review numerical defaults and training/build GPUs; the
existing export helper requires a separate inference GPU of matching capability.
The shared core requires `warmup_episodes * iterations_per_ep == replay_buffer_size`
in `po4ao.conf`. Preserve other runtime geometry/gains/limits. Change its engine path:

```ini
[parameters]
engineDirs=/data/po4aoTrainer/current
engineName=policy.trt
```

Do not create `current` as a directory; publication creates an atomic symlink.
Restart the named C++ runtime after editing its configuration: `xctrl restart po4ao`.

**Register/connect:** add `po4aoTrainer po4aoTrainer` to `proclist_RTC.txt`, append
`po4aoTrainer` to `isRTC.conf`'s `[local] drivers`, and expose it through the same GUI
aggregation as step 1. Apply only the affected server configurations. `xindiserver`
creates its driver link; XDevice creates its FIFOs, logs, telemetry and PID record
using the existing MagAO-X permissions/PID helper. Ensure the runtime account has
normal access to `/opt/MagAOX/drivers/fifos`, shared memory and the GPUs.

Before startup, run as the intended runtime account from an unrelated directory,
without personal `PYTHONPATH`:

```sh
cd /tmp
env -u PYTHONPATH /opt/conda/envs/po4aoTrainer/bin/python -m xapp.po4aoTrainer.check_deployment
/opt/MagAOX/bin/po4aoTrainer -h
xctrl startup po4aoTrainer
xctrl status po4aoTrainer
xctrl peek po4aoTrainer
```

The checker validates imports/config/paths/atomic publication, not GPU execution.
The same GUI now enables managed training controls; discovery itself starts no work.
Use **Reset → wait for completion → Enable PO4AO (fallback) → wait for collection
Ready at its full frame target → Start learning**. Reset leaves control Off, so
Start alone does not acquire initial warmup. Enable selects fallback and starts its
initial exploratory collection; keep that selection through collection. Reverify
the manual downstream route above after any runtime/configuration change. Then
verify a real export loads, iteration increases, and the active ID matches publication.
Pause/Resume must preserve the worker/models; Disable must pause learning and select
integrator; Enable restores the latest eligible policy without resuming learning.
Recollect preserves the controller, adds exploration and resumes only previously
running learning. Reset stops the worker and clears retained eligibility. The app
writes metrics and diagnostic snapshots for notebook plotting.
Test trainer loss/arrival while another client has an outstanding operation:
commands must not bypass it through the other route. Verify delayed publication,
Reset/Disable from another client and status expiry/recovery before accepting
managed operation. Measure inference latency during training and export.

Valid hot training-config edits apply at iteration boundaries; architecture/history/
normalization/GPU changes require Reset, shown in the GUI. No app restart is needed.
For NN controls, if this branch's NN version is not installed, build with
`make -C "$MAGAOX_SRC/apps/nnReconstructor"`, stop the configured NN process with
`xctrl shutdown <NN-device>`, install with
`make -C "$MAGAOX_SRC/apps/nnReconstructor" install`, then
`xctrl startup <NN-device>`. Use the actual device name from the app config. Verify
the configured milk FPS writer identity and linked modulation preset (2000 Hz,
radius 3) before testing handover. No `dlDataCollection` change is needed here.

## Recovery and completion

On Alert, inspect actual state/logs before retrying. Missing frames can leave Reset
pending. A supervisor restart loses in-memory optimizer state and requires Reset;
old engine files alone do not restore eligibility. Switching back to notebook use
means stopping managed training/export, stopping the trainer, and restoring the
notebook engine path/restarting C++ if it differs. The same GUI remains usable.

Rollback: stop the named trainer, restore the saved runtime binary/config and prior
source versions, then restart only affected processes. Manually restore the mfilt
input and calibration/setup recorded before the trial using the normal routing
and calibration commands; do not assume a restoration target. Verify the effective
input stream and advancing consumption counters again after both changes, then
select the intended controller yourself. Stopping the trainer alone does not stop
C++ inference.
Record target builds, successful export/reload, button results and inference latency.
Commit/push software and config revisions before scientific use, following the
[MagAO-X development guide](https://magao-x.org/docs/handbook/software/hacking.html).

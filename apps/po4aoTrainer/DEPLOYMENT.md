# PO4AO deployment: existing notebook, then managed training

Use the same `po4aoGUI` for both stages. You manage loop state and run the commands
below. The source checkout on exao2 is now `/home/rlandman/MagAOX`, branch
`rlandman/nn-po4ao-gui` (includes fixes through `be3f7652`). Installed apps are unchanged.

Read-only file checks on **2026-11-03** confirmed: `po4ao` is already registered in
`proclist_RTC.txt` and `isRTC.conf`; its output is `po4ao_output`; its notebook engine
is `/srv/aoc/home/rlandman/data/PO4AO/po4ao_model.trt`. The notebook uses `torchpy`
(Python 3.11). Neither the trainer nor new GUI is installed. exao2 has no qmake on PATH;
build the GUI on your usual MagAO-X GUI host with Qt/mxlib available.

## 1. Install PO4AO on exao2

As `rlandman`, save the installed binary/configuration and build only this app:

```sh
cd /home/rlandman/MagAOX
git status --short --branch
BACKUP="$HOME/po4ao-backup-$(date +%Y%m%d-%H%M%S)"
mkdir "$BACKUP"
cp /opt/MagAOX/bin/po4ao "$BACKUP/"
cp -L /opt/MagAOX/config/{po4ao.conf,isRTC.conf,proclist_RTC.txt} "$BACKUP/"
readlink -f /milk/shm/aol1_mfiltINPUT.im.shm > "$BACKUP/mfilt-input.txt"
cat /milk/shm/aol1_calib_source.txt > "$BACKUP/calibration.txt"
make -C apps/po4ao clean && make -C apps/po4ao && \
  xctrl shutdown po4ao && make -C apps/po4ao install && xctrl startup po4ao
xctrl status po4ao
xctrl peek po4ao
```

Keep the backup path and verify the recorded route/calibration identify the previous
setup; record them manually if either file is unavailable. Keep `po4ao.conf`
unchanged for notebook testing. CUDA/TensorRT dependencies must build successfully
before stopping the old app. No NN, dlDataCollection or INDI-server reinstall is needed.

## 2. Install the GUI on the GUI host

Use a personal checkout of the same branch. For a new checkout, fetch it directly
from exao2; this does not require the branch to be on GitHub:

```sh
git clone --single-branch --branch rlandman/nn-po4ao-gui \
  rlandman@exao2:/home/rlandman/MagAOX "$HOME/MagAOX-po4ao-gui"
cd "$HOME/MagAOX-po4ao-gui"
make -C INDI/liblilxml && make -C INDI/libcommon && \
  make -C gui/apps/po4aoGUI && make -C gui/apps/po4aoGUI install && \
  po4aoGUI po4ao po4aoTrainer
```

Reuse that checkout if it already exists; save any previous `/usr/local/bin/po4aoGUI`
before installing. The host's existing localhost:7624 aggregation must expose
`po4ao`. If absent, append `po4ao@<existing-RTC-tunnel-name>` to its `[remote] drivers`
and restart only that aggregator. Its configuration was not inspected on exao2.

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
GUI changes neither training nor inference. NN controls require the companion.

## 4. Add managed training later, on exao2

Stop notebook training/export. The managed trainer starts its own models/optimizer;
it does not adopt the notebook's session. Clone the known notebook environment into
a shared **new** environment; leave the original untouched. `CacaoProcessTools` is
missing for Python 3.11 (installed copies are 3.10/3.13). Build just that module with
milk's existing CMake target, link installed milk libraries, and install into the
trainer environment. These target build commands still require operator validation:

```sh
cd /home/rlandman/MagAOX
sudo /opt/conda/bin/conda create -y -n po4aoTrainer --clone /opt/conda/envs/torchpy
TRAIN_PY=/opt/conda/envs/po4aoTrainer/bin/python
sudo -H "$TRAIN_PY" -m pip install pybind11
CPT_BUILD="$HOME/build-po4ao-cpt311"
PATH="/opt/conda/envs/po4aoTrainer/bin:$PATH" cmake \
  -S /opt/MagAOX/source/milk/python_module -B "$CPT_BUILD" \
  -DPYTHON_EXECUTABLE="$TRAIN_PY" \
  -DCMAKE_CXX_FLAGS="-I/usr/local/milk/include -I/usr/local/milk/include/CommandLineInterface -I/usr/local/milk/include/ImageStreamIO" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L/usr/local/milk/lib" \
  -DCMAKE_BUILD_RPATH=/usr/local/milk/lib && \
  cmake --build "$CPT_BUILD" --target CacaoProcessTools && \
  sudo install -m 755 "$CPT_BUILD"/CacaoProcessTools.cpython-311-*.so \
    /opt/conda/envs/po4aoTrainer/lib/python3.11/site-packages/
sudo -H "$TRAIN_PY" -m pip install \
  'xconf @ git+https://github.com/xwcl/xconf.git@d6f4de40bd290510bfe191b777d85d36b467d0ce' \
  'purepyindi2 @ git+https://github.com/xwcl/purepyindi2.git@c26a8276c95b95aef145c50439e291b782eab190'
sudo -H "$TRAIN_PY" -m pip install ./magaox-python
make -C apps/po4aoTrainer INSTRUMENT_CONDA_ENV=po4aoTrainer install
cp -i apps/po4aoTrainer/configs/{po4aoTrainer.conf,po4ao_training.conf} /opt/MagAOX/config/
sudo install -d -o xsup -g "$(id -gn xsup)" -m 2755 \
  /data/po4aoTrainer /opt/MagAOX/logs/po4aoTrainer /opt/MagAOX/telem/po4aoTrainer
```

Skip environment creation if it already exists. Review the two config files:

- `po4aoTrainer.conf`: actual device names/FPS writer, `artifact_root="/data/po4aoTrainer"`,
  and `source_revision="rlandman/nn-po4ao-gui@<git rev-parse HEAD>"`. The configured
  TensorRT builder exists at `/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec`.
- `po4ao_training.conf`: training settings and GPU assignments. Current defaults use
  GPU 1 for training/building, with runtime inference on GPU 0; verify compatibility.
- Existing `po4ao.conf`: keep geometry/gains/limits. Its current `10 * 1000 == 10000`
  warmup/replay dimensions already satisfy the trainer. Change only the engine path:

```ini
[parameters]
engineDirs=/data/po4aoTrainer/current
engineName=policy.trt
```

Publication creates `current`; do not make it a directory. Add
`po4aoTrainer po4aoTrainer` to `proclist_RTC.txt` and append `po4aoTrainer` to
`isRTC.conf`'s `[local] drivers`. Expose it through the GUI host's aggregation as in
step 2, then restart only affected INDI servers (`xctrl restart isRTC` on exao2).

Run the checker as the runtime account (`xsup`); then start the apps:

```sh
cd /tmp
sudo -u xsup env -u PYTHONPATH /opt/conda/envs/po4aoTrainer/bin/python -m xapp.po4aoTrainer.check_deployment
xctrl restart po4ao
xctrl startup po4aoTrainer
xctrl status po4aoTrainer
xctrl peek po4aoTrainer
```

In the same GUI: **Reset → wait → Enable fallback → collection Ready/full target →
Start learning**. Start alone does not acquire initial warmup. Reverify downstream
routing after configuration changes. Confirm real export/reload and matching model ID,
then test Pause/Resume, Disable/Enable, Recollect and Reset. Disable pauses learning;
Enable restores the retained policy without resuming training. Recollect preserves
the controller and resumes only previously running learning. Reset stops the worker.

Valid hot config edits apply during training; reset-only changes are shown in the
GUI. Before accepting managed operation, test delayed loads, other-client operations,
trainer loss/return and status recovery, and measure latency during training/export.
NN switching also needs the branch's NN app installed; if necessary build/install
only `apps/nnReconstructor` between named shutdown/startup, and verify its configured
writer and linked CACAO modulation (2000 Hz, radius 3).

## Recovery

On Alert, inspect readback/logs before retrying. Missing frames can leave Reset
pending; a lost managed worker session requires Reset. To return to the notebook,
stop managed training/export and `xctrl shutdown po4aoTrainer`, restore the saved
notebook engine path, and restart `po4ao`.

For full rollback, stop the affected apps, restore the saved configs and install the
saved binary with `sudo install -m 4755 -o root "$BACKUP/po4ao" /opt/MagAOX/bin/po4ao`;
restart only affected processes/INDI servers. Manually restore the **recorded** mfilt
input and calibration, then verify consumption again. Restore the old GUI if replaced.
Stopping the trainer alone does not stop C++ inference.

Target builds, permissions, GPU export/reload, live INDI/FPS, routing and timing remain
operator acceptance checks. Commit/push software and configuration before scientific
use, following the [MagAO-X development guide](https://magao-x.org/docs/handbook/software/hacking.html).

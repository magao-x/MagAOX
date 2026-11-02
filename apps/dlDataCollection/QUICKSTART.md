# dlDataCollection: collect with another ND filter

Reuse the existing coefficient bank when you want the same command distribution.
Change the physical `fwtelsim` filter and the output directory. No new bank,
config edit, or app restart is needed for each filter.

This procedure assumes the updated app is already running and visible through
INDI. Run the commands yourself in an **exao2/RTC terminal**, normally as `xsup`.
Use the established internal-source calibration setup, with the relevant AO loop
open and the collector's dedicated DM channel initially zero and exclusively
available to it. Start applies the bank's commands to that channel.

## 1. Finish the previous collection and change the filter

```sh
getINDI -h localhost -p 7624 -t 3 'dlDataCollection.collection.*'
```

Wait for `phase=complete` and an empty `error`. To interrupt a run instead:

```sh
setINDI -h localhost -p 7624 'dlDataCollection.stop.request=On'
```

Wait for `phase=cancelled` before continuing. If the phase is `failed`, read
`collection.error` and resolve it first. A fresh app starts at `phase=idle`.

In cursesINDI, select the desired **fwtelsim** ND filter using its named preset.
The updated collector checks the wheel and `flipwfsf` readbacks automatically
before Start. It records their positions in each run's `run.conf`; it does not
move either device. Directory names are convenient labels, while the recorded
`nd` and `flipwfsf` values describe the reported instrument state.
It also reads `camwfs.fps.current` and `camwfs.emgain.current`, saves them as
`camwfs_fps` (Hz) and `camwfs_emgain` (gain factor), and displays both under
`dlDataCollection.observation`. Finish or Stop before changing either camera
setting: reported changes or invalid/stale readbacks abort the run while
preserving its partial data.

## 2. Select the output directory and reuse the bank

Edit these example paths. `BANK` can still end in `ND1` when collecting at ND2:

```sh
BANK=/home/rlandman/NNReconstructor_2026B/banks/20261015/ND1
OUTPUT=/data/calib/nnReconstructor/20261021/ND2

setINDI -h localhost -p 7624 "dlDataCollection.bank.target=$BANK"
setINDI -h localhost -p 7624 "dlDataCollection.output.target=$OUTPUT"

getINDI -h localhost -p 7624 -t 3 \
    'dlDataCollection.bank.current' 'dlDataCollection.output.current'
```

**Check both readbacks before Start.** Bank/output changes are accepted only
while the collector is available for a new run. The bank must contain
`manifest.conf` and be readable by the app account; the output parent must be
writable by that account. The app creates missing output directories if allowed.

Do **not** add `-s` to the bank/output commands: these properties are text,
whereas `-s` means switch and can cause the request to be silently ignored.
Runtime path changes are not saved back into the config; after an app restart,
check the paths again.

## 3. Start and monitor

```sh
setINDI -h localhost -p 7624 'dlDataCollection.start.request=On'
getINDI -h localhost -p 7624 -t 3 'dlDataCollection.collection.*'
```

Repeat the query, or watch `dlDataCollection` in cursesINDI:

| Field | Meaning |
|---|---|
| `phase` | `preparing`, `collecting`, `waiting_io`, `draining`, then `complete` |
| `saved` | Accepted image/command pairs |
| `written` | Complete sets written to disk |
| `directory` | Actual run directory; record this for loading/training |
| `error` | Failure explanation; normally empty |
| `torn` / `duplicates` | Rejected unstable snapshots / repeated frame notifications |

Start can first show `checking_observation` for up to five seconds. Both devices
must be visible through the collector's INDI server, settled, and reporting a
known position. Current readbacks appear under `observation.nd` and
`observation.flipwfsf`. If either changes or becomes unavailable during capture,
the collector stops and preserves the partial run instead of mixing labels.

Wait for **`phase=complete`**, not just the last increase in `saved`: file writing
may still be draining. Each Start creates a new `run_*` subdirectory inside
`OUTPUT`; existing runs are not overwritten. Complete sets have
`complete_dataset_N.conf` markers, and `finished.conf` records the run status.
Interrupted runs can contain partial sets; do not treat these as complete sets.

For ND3, repeat from step 1 and change `OUTPUT` to the ND3 directory. Keep `BANK`
unchanged to reuse the distribution. Keep the existing validated camera, crop,
normalization, and `NumFrameSkip` settings unless deliberately recalibrating.

See the [full README](README.md) for configuration and file layouts, and the
instrument handbook for [cursesINDI](https://magao-x.org/docs/handbook/operating/software/guis/cursesINDI.html)
and [setINDI](https://magao-x.org/docs/handbook/operating/software/utils/setINDI.html).
The procedure matches the repository source; the installed RTC version and live
instrument state have not been checked here.

For new banks, the [notebook update](README.md#notebook-defaults-and-automatic-training-labels)
defaults to `banks/YYYYMMDD/BANK_NAME` and keeps your recipe and full-path override.
That section also explains the small training-loader update for automatic ND
labels and filtering by `flipwfsf`.

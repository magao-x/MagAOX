# ICC cpuset inventory

## Request

We need to update cpusets on ICC.  Please see plans history and script for RTC, including for removing cpusets then making new ones.

/opt/MagAOX/config is up to date on this laptop.  Please start by using proclist_ICC.txt to determine running apps, and the config for those apps to determine used cpusets.

- Note that acc001 is not currently used, so we can consider its assigned cpusets unused.
- The mfilt cpuset on ICC can be turned into a spare.

With that info, make table of cpuset assignments on ICC and place it in this doc.  After that we'll decide what to do for reconfiguration.

## Status and decisions, updated 2026-11-15

The operator approved moving ICC CACAO's `mfilt` to `acqu_mvm`, disabling the
unused `acc001` process-list entries, and allocating three spare cores to
`camscc` plus one to `camscc_sw`. The operator also confirmed that `RTmon` is
used by `milk-shmimmon`, and that `shmimTCP` is an intermittent CACAO network
transfer process using CPU58. Sharing its L3 with the camera is acceptable for
now.

Workstation changes use CPUs **56,57,59 for `camscc`** and **55 for `camscc_sw`**.
The **`dm02disp09_sw` recorder moves from CPU59 to CPU61**, using inactive
`acc001`'s former spare. This gives the camera three cores in the operator-confirmed
L3 group `56-59`, separate from `acqu_mvm` on CPU45 in `44-47`. CPU55 was
`spare55`. Remaining spares are **44,46,47**, including the core released by
moving `mfilt-2` to `acqu_mvm` on CPU45. Existing app configs retain their cpuset
names, and the IRQ assignments are unchanged.

For the push, the operator reverted the `camscc.conf` and `camscc-sw.conf`
changes because those files have changed substantially on ICC. The new cpusets
are reserved by the setup script, but enabling their two `[framegrabber].cpuset`
settings is deferred until after deployment using the current ICC files. Neither
camera config is included in the config commit.

The first workstation layout used `44,46,47` for `camscc`. The operator then
requested separation from `acqu_mvm`'s cache group. The move to `56,57,61` used
only spare cores and changed no other active assignment or app/CACAO config.
Its initial cache rationale extrapolated RTC's eight-core L3 groups; the
operator's 2026-11-15 identification of ICC as a **3995WX** supersedes that
assumption. The subsequent ICC readback confirms four-core sharing groups, so
the final revision swaps the camera's CPU61 with the recorder's CPU59. This
moves one other active cpuset and keeps the camera within one L3 cache.

### ICC processor and confirmed cache membership, 2026-11-15

The operator identifies ICC as an **AMD Ryzen Threadripper PRO 3995WX 64-Cores**.
This is Zen 2: four cores share a 16 MB L3 in each core complex (CCX), with two
CCXs per eight-core chiplet (CCD). RTC's 5995WX uses Zen 3 with a unified
eight-core L3. The CPU model establishes that architectural distinction, but
does not establish ICC's Linux CPU numbering, cache IDs, or NUMA configuration.
See [AMD's Zen 2/Zen 3 cache sizes, slide 14](https://gpuopen.com/download/GDC_AMD_Ryzen_Processor_Software_Optimization.pdf#page=14)
and [NASA's description of Zen 2 versus Zen 3 cache complexes](https://www.nas.nasa.gov/hecc/support/kb/pdf-cat/176/).

The operator ran the L3 `shared_cpu_list` readback on **`exao3` (ICC)** and
provided the following output on 2026-11-15:

```text
CPU45 L3 shared CPUs: 44-47
CPU55 L3 shared CPUs: 52-55
CPU56 L3 shared CPUs: 56-59
CPU57 L3 shared CPUs: 56-59
CPU58 L3 shared CPUs: 56-59
CPU59 L3 shared CPUs: 56-59
CPU60 L3 shared CPUs: 60-63
CPU61 L3 shared CPUs: 60-63
CPU62 L3 shared CPUs: 60-63
CPU63 L3 shared CPUs: 60-63
```

These are observed cache-sharing lists. The workloads below describe the
intended placement after deployment and the deferred camera config edits,
not runtime placement measured on ICC.

| Confirmed L3 shared CPUs | Configured occupants after migration |
|---|---|
| `44-47` | `acqu_mvm`45 and spares44/46/47 |
| `52-55` | DM recorder52, average fits53/54, `camscc_sw`55 reserved pending camera config edit |
| `56-59` | `camscc`56/57/59 reserved pending camera config edit; intermittent CACAO `shmimTCP` process and configured NIC IRQ on58 |
| `60-63` | DM recorders60/61, `aol2` writer/buffer62/63 |

All three reserved camera cores share L3 and exclude CPU45's L3. CPU58 in the same
L3 is used intermittently by the CACAO `shmimTCP` network-transfer process, as
confirmed by the operator. The ICC-RTC NIC IRQ is also routed to CPU58 by the
setup script. The operator accepts this sharing for now. `camscc_sw` on CPU55
uses the separate `52-55` L3. The readback does not establish PCIe/DMA locality,
NUMA masks, actual thread affinity, or timing performance.

Both repositories use feature branch `jrmales/icc-cpusets`: MagAOX is based on
`dev` at `5d592755`, and config is based on refreshed `onsky` at `93104c5`.
The original config inventory used `f8248d9`. While preparing the push, upstream
commit `1666861` was found to contain the identical NCPC `mfilt` move to
`acqu_mvm`/45, so the new config commit only comments out the three `acc001`
entries. All agent work was done on workstation `mx11`. Instrument deployment
and runtime verification remain operator tasks; agents performed no instrument
interrogation or cpuset/IRQ commands.

The table below is the **updated configured ICC assignment map**, using
uncommented process-list entries as the requested app roster. It is not a measurement of running
processes or live affinity. ICC and RTC have independent CPU/cpuset namespaces.

### Sources and coverage

- [ICC process list](/opt/MagAOX/config/proclist_ICC.txt) and exact, case-sensitive
  `/opt/MagAOX/config/<process-ID>.conf` files. Commented-out settings are excluded.
- [ICC make_cpusets](../../../rtSetup/ICC/make_cpusets) for cpuset names and CPUs.
- [NCPC CACAO setup](/opt/MagAOX/config/cacao/ncpc/fpssetup.setval.ncpc.conf) and
  [NCPC variables](/opt/MagAOX/config/cacao/ncpc/cacaovars.ncpc.bash) for processes
  launched separately from the app roster: loop number `2`, DM index `02`.
- [ICC move_irqs](../../../rtSetup/ICC/move_irqs),
  [irqbalance settings](../../../rtSetup/ICC/irqbalance), and
  [IRQ policy](../../../rtSetup/ICC/irqbalance_policyscript) for dedicated IRQ uses.
- [RTC restructuring history](../2026-10/rtc-cpuset-restructuring.md), including
  its removal/recreation procedure, and
  [RTC remove_cpusets](../../../rtSetup/RTC/remove_cpusets).
- Operator-provided ICC L3 readback from `exao3`, reproduced above.
- Operator clarification that CACAO `shmimTCP` intermittently uses CPU58 for
  network transfers, and acceptance of sharing the camera's L3 for now.

The initial inventory had 104 entries and 98 exact configs, including the three
unused `acc001` entries. After commenting those entries, with the two camera
config edits deferred, the roster contains **101 entries**, **95 exact configs**,
and **20 assigned configs / 25 explicit settings / 20 distinct sets**. The other
**75 exact configs** have no nonempty cpuset setting. Six exact instance configs
are absent, as detailed below. Enabling the two camera assignments after
deployment would give 22 assigned configs / 27 settings and 73 without an
assignment for this snapshot. The global `magaox.conf` is empty. These counts
exclude CACAO, manual monitors, and IRQ assignments, which are included below.

### ICC CPU and cpuset assignments

The setup script assigns `system` to CPUs `0-31,64-95`, provides 30 exclusive
sets covering CPUs `32-63`, and offlines CPUs `96-127`. `camscc` spans three
cores; each other dedicated set spans one. Every dedicated set has
`cpu_exclusive=1` and `mems=0`. The operator-confirmed model has 64 cores/128
threads. Cache membership for the sampled CPUs is confirmed above; the readback
does not independently verify every SMT sibling pair or the memory-node masks.

In the table, **FG** means `[framegrabber].cpuset`, **SM** means
`[shmimMonitor].cpuset`, and **DM** means `[dm].cpuset`, in the named app's exact
config. These place individual threads, not necessarily the whole process.
CACAO rows use `procinfo.cset` and the matching numeric `procinfo.taskset`.
IRQ entries describe the routing script's assignments, not observed IRQ activity.

| ICC CPU(s) | Current cpuset | Configured app threads / CACAO processes | Other consumers / planning status |
|---|---|---|---|
| `0-31,64-95` | `system` | Inherited/background placement; see exceptions below | Default IRQ pool, including NVIDIA IRQs since 2026-11-15; 32 cores with both hardware threads enabled |
| 32 | `dm02comb` | CACAO `DMch2disp-02` | NCPC DM combiner |
| 33 | `ncpc` | `dmncpc` DM | Command-monitor thread; no explicit SDK/post-worker cpusets |
| 34 | `camflowfs` | `camflowfs` FG | `pvcamPCIE_1` IRQs |
| 35 | `camflowfs_sw` | `camflowfs-sw` FG | Recorder acquisition thread |
| 36 | `camllowfs` | `camllowfs` FG | `pvcamPCIE_0` IRQs |
| 37 | `camllowfs_sw` | `camllowfs-sw` FG | Recorder acquisition thread |
| 38 | `camsci1` | `camsci1` FG | Camera NIC `enx503eaa0ceeff` IRQs |
| 39 | `camsci1_sw` | `camsci1-sw` FG | Recorder acquisition thread |
| 40 | `camsci2` | `camsci2` FG | Camera NIC `enx503eaa0cf4cd` IRQs |
| 41 | `camsci2_sw` | `camsci2-sw` FG | Recorder acquisition thread |
| 42 | `camflowfs_fit` | `camflowfs-fit` SM + FG | Both threads share the set |
| 43 | `camllowfs_fit` | `camllowfs-fit` SM + FG | Both threads share the set |
| 44 | `spare44` | No consumer found in reviewed sources | Spare |
| 45 | `acqu_mvm` | CACAO `acquWFS-2`, `wfs2cmodeval-2`, `mfilt-2`, `mvalC2dm-2` | NVIDIA IRQs no longer routed here (2026-11-15, see `agents/plans/2026-10/rtc-cpuset-restructuring.md`) |
| 46 | `spare46` | No consumer found in reviewed sources | Spare; released by moving `mfilt-2` to CPU45 |
| 47 | `spare47` | No consumer found in reviewed sources | Spare |
| 48 | `mlat` | CACAO `mlat-2` | Configured latency-measurement process; retain reservation pending decision |
| 49 | `RTmon` | `milk-shmimmon` | Confirmed by operator; retain reservation |
| 50 | `dm02disp03_sw` | `dm02disp03-sw` FG | Recorder acquisition thread |
| 51 | `dm02disp04_sw` | `dm02disp04-sw` FG | Recorder acquisition thread |
| 52 | `dm02disp08_sw` | `dm02disp08-sw` FG | Recorder acquisition thread |
| 53 | `camflowfs_avg_fit` | `camflowfs-avg-fit` SM + FG | Both threads share the set |
| 54 | `camllowfs_avg_fit` | `camllowfs-avg-fit` SM + FG | Both threads share the set |
| 55 | `camscc_sw` | Reserved for `camscc-sw` FG; config edit deferred | Recorder acquisition thread; replaces `spare55` |
| `56-57,59` | `camscc` | Reserved for `camscc` FG; config edit deferred | Three camera cores in L3 `56-59`; former CPU59 recorder moves to61 |
| 58 | `shmimTCP` | Intermittent CACAO network-transfer process; operator-confirmed | ICC–RTC NIC `enx98b78501ae64` IRQs also routed here; L3 sharing with `camscc` accepted for now |
| 60 | `dm02disp_sw` | `dm02disp-sw` FG | Recorder acquisition thread |
| 61 | `dm02disp09_sw` | `dm02disp09-sw` FG | Recorder acquisition thread; moved from59 into former `spare61` |
| 62 | `spare62` | `aol2_modevalWFS-sw` FG | **In use despite spare name** |
| 63 | `spare63` | `aol2_modevalWFS_cbuff` SM + FG | **In use despite spare name** |
| `96-127` | None | Offline per setup script | Disabled hardware-thread siblings of dedicated CPUs |

This leaves **three spare CPUs: 44,46,47**. Four cores are reserved for the
deferred camera assignments; the other 25 dedicated CPUs have configured app,
CACAO, IRQ, or operator-confirmed monitor uses. This accounting does not
establish runtime occupancy.

### Config interpretation and exceptions

- The three `acc001` entries, `acc001-cbuff`, `acc001-psds`, and `acc001-sw`, are
  now commented out in `proclist_ICC.txt`. Their configs are retained. Only
  `acc001-cbuff` has explicit cpuset settings (`spare61`, twice). This name is
  now retired; reconfigure that inactive instance before any future reactivation.
  `acc001-sw` has a **commented-out** `cpuset=spare36`; it does not reserve CPU36.
  The separate `acc001.conf` belongs to an instance absent from the ICC roster
  and is not an ICC allocation.
- `camscc` and `camscc-sw` retain their commented-out FG cpuset settings in this
  config branch. The operator will enable them in the current ICC files after
  deployment, selecting the camera FG thread and recorder acquisition thread.
  The `orcaCtrl` source is absent here, so placement of its SDK/helper threads
  is not established by this change. `camngws` still has only a commented-out
  cpuset setting.
- The listed `lopredctrl` uses [lopredctrl.conf](/opt/MagAOX/config/lopredctrl.conf),
  which has no cpuset setting. The differently cased, unlisted `loPredCtrl.conf`
  sets SM and FG to `acqu_mvm`; those entries must not be attributed to
  `lopredctrl`. `hwpsequence` explicitly leaves its SM cpuset empty.
- `dm02disp01-sw`, the camera dark/average integrators, `lopsds`, `logopt`, and
  both `mzmqServerICC` instances have no explicit cpuset settings. In particular,
  neither the CACAO `shmimTCP` process nor the NIC IRQ assignment establishes
  that either mzmq server is pinned to CPU58.
- All listed `streamWriter` assignments are to their acquisition/FG threads.
  Their separate `[writer].cpuset` settings are absent, so the disk-writing
  threads retain inherited placement. The frameGrabber/shmimMonitor helpers
  also default to an empty cpuset. Empty settings do not establish a dedicated
  CPU; under the intended startup flow they ordinarily inherit `system`.
- `dmncpc` only sets the DM monitor cpuset. The BMC SDK/post-worker cpuset
  defaults are empty; do not assign those workers to CPU33 based solely on
  `[dm].cpuset=ncpc`.
- CACAO now sets `mfilt-2.procinfo.cset=acqu_mvm` and `taskset=45`, retaining
  `RTprio=48` and enabled `CACAO_FPSPROC_MODALFILTERING`. Both placement settings
  were updated together; the filtering process remains enabled.
- IRQ policy comments reversed the `pvcamPCIE_0`/`pvcamPCIE_1` camera labels
  relative to `move_irqs`. The comments now follow `move_irqs` (2026-11-15).
  The table follows the executable routing statements; actual device
  identities should still be confirmed on the ICC before changing that routing.

| Missing exact instance config | Interpretation |
|---|---|
| `icc_aoc_indi.conf`, `icc_rtc_indi.conf` | `sshDigger` uses the shared `sshTunnels.conf`; these missing per-instance files are expected. No explicit cpuset there. |
| `coralign.conf` | No exact local config; differently cased `corAlign.conf` is not assumed to apply. No dedicated assignment established. |
| `pyexample.conf` | No exact local config for the listed `pythonIndiExample` instance. |
| `stageqwprot1.conf`, `stageqwprot2.conf` | No exact local configs for these listed `elliptecCtrl` instances. |

The last four missing configs remain inventory gaps; no CPU reservation is
inferred from their absence. Manual launches and deployed overrides are outside
this workstation snapshot. No nonempty cpuset defaults were found in the
available listed app sources. Exact executable source directories are absent
here for `coralign`, `elliptecCtrl`, `fpmCtrl`, `hwpSequencer`, and `orcaCtrl`
(`apps/corAlign` exists with different capitalization), so their source defaults
were not established by this review.

### Migration and operator verification

The RTC migration stopped app/CACAO consumers and manual monitors, previewed
cleanup, removed verified empty cpusets, recreated the new layout, applied the
matching IRQ configuration, and then restarted and verified placement. Its
cleanup helper preserves root/`system`, rejects unknown or occupied sets, and
never kills or moves tasks. Re-running `make_cpusets` alone does not remove old
exclusive sets or apply IRQ routing.

The ICC cleanup entrypoint shares the RTC implementation, with explicit role
markers in the sibling setup scripts and separate retired-name allowlists. ICC
recognizes its current sets plus the legacy names `spare44`, `mfilt`, `spare47`,
`spare55`, `spare56`, `spare57`, and `spare61`. This supports cleanup from the
original layout or the initial camera allocation. RTC-only names are not
accepted merely because the implementation is shared.
`make rtscripts_install MAGAOX_ROLE=ICC` installs the helper alongside
the ICC setup script. Deploy the helper and marked setup script together;
an older setup without the role marker is rejected before inspecting cgroups.

The operator has completed the cache-membership check for the final allocation.
For future rechecks, the read-only command used was:

```bash
for cpu in 45 55 56 57 58 59 60 61 62 63; do
    for cache in /sys/devices/system/cpu/cpu"$cpu"/cache/index*; do
        if [ "$(cat "$cache/level")" = 3 ]; then
            printf 'CPU%s L3 shared CPUs: ' "$cpu"
            cat "$cache/shared_cpu_list"
        fi
    done
done
```

After deploying the matching scripts and configs to ICC, the operator should:

1. Stop the MagAOX apps, CACAO processes, and `milk-shmimmon`, and prevent
   automatic restarts during migration. Retain the old scripts/configs for rollback.
2. Preview and remove empty groups from the updated source checkout:

   ```bash
   sudo python3 rtSetup/ICC/remove_cpusets --dry-run
   sudo python3 rtSetup/ICC/remove_cpusets --apply
   ```

   Resolve any occupied or unknown groups before rebuilding. The old exclusive
   sets overlap the new camera sets, so do not simply rerun `make_cpusets`.
3. Recreate the layout and reapply the existing ICC IRQ routing:

   ```bash
   sudo bash rtSetup/ICC/make_cpusets
   sudo bash rtSetup/ICC/move_irqs
   ```

   System CPUs, offline siblings, IRQ targets, and irqbalance policy are unchanged.
4. In the current ICC configs, enable `[framegrabber] cpuset=camscc` in
   `camscc.conf` and `[framegrabber] cpuset=camscc_sw` in `camscc-sw.conf`,
   preserving the other deployed settings. These edits are deliberately omitted
   from the pushed config branch to avoid conflicts with ICC's newer files.
5. Apply/reload the updated NCPC FPS setup as part of restarting CACAO; changing
   its config file alone does not update an existing FPS instance. Restart the
   apps and monitor, leaving `acc001` disabled.
6. Verify `camscc` CPUs `56-57,59`, `camscc_sw` CPU55, `dm02disp09_sw` CPU61,
   and `mfilt-2` cpuset `acqu_mvm` / taskset45. Check camera FG/SDK/helper and
   recorder thread affinity,
   `milk-shmimmon` on CPU49, and camera/loop timing under load. SDK/helper placement
   requires runtime confirmation because its source is not in this checkout.

### Changed files and validation

MagAOX changes:

- `rtSetup/ICC/make_cpusets`: new camera sets, recorder relocation, and documented
  monitor use and confirmed L3 groups.
- `rtSetup/ICC/remove_cpusets`, `rtSetup/RTC/remove_cpusets`, and
  `rtSetup/RTC/tests/remove_cpusets_test.py`: shared removal with separate
  role-specific ownership. The ICC source entrypoint is a relative symlink;
  installation copies the implementation and uses the installed ICC setup.
- `rtSetup/RTC/make_cpusets`: role metadata for cleanup; RTC assignments unchanged.
- This plan: decisions, updated inventory, and migration procedure.

Config changes and dependencies:

- New commit: `proclist_ICC.txt`, commenting all three unused `acc001` entries.
- Already in the refreshed base at `1666861`:
  `cacao/ncpc/fpssetup.setval.ncpc.conf`, with `mfilt` cset and numeric affinity.
- Deferred, excluded from the push: `camscc.conf` and `camscc-sw.conf` FG
  cpuset settings, to be enabled against the current ICC files after deployment.

Workstation validation passed:

- All 17 temporary-filesystem/mock cleanup tests, covering both roles, retired
  names, unknown/occupied sets, default dry-run, role markers, and installed links.
  Run with `python3 -B rtSetup/RTC/tests/remove_cpusets_test.py`.
- `bash -n` for both setup scripts, plus whitespace checks in both repositories.
  The touched code is Bash/Python; clang-format does not support these languages,
  so their existing script formatting is retained.
- A static map/config audit: 30 exclusive sets cover all 32 dedicated CPUs once,
  without overlap with `system`; all unchanged sets retain their original masks.
  All 25 active app cpuset keys, six CACAO cset/taskset pairs, and IRQ cpuset
  references resolve. The table and source links match the resulting files.

The 2026-11-15 operator readback confirms cache separation for the final
`camscc=56,57,59` / `dm02disp09_sw=61` swap. Only the setup script and this plan
change for that revision; the existing cleanup names and app/IRQ configs remain
valid. Agents performed no hardware tests or live cgroup operations. Runtime
camera/recorder thread placement and timing after the migration remain operator
verification items.

Push preparation preserves the camera config reverts, includes this engineering
record with the functional MagAOX changes, and follows with setup-comment
documentation cleanup. Unrelated untracked `xlog/` content is excluded.

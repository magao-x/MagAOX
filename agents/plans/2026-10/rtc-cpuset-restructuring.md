# RTC cpuset inventory and restructuring plan

Status: baseline inventory prepared 2026-10-26 and refreshed from the operator's
synced configs for `po4ao`, `nnReconstructor`, `dmtweeter-synco` and the newly
listed `dlDataCollection`. Remaining deployed/runtime differences await operator
confirmation. Operator topology readback confirms the core/cache groups below.
The annotated table and subsequent corrections authorize the first rename
migration, EDT IRQ placement on CPU34 and synco on CPU41. Workstation
scripts/configs are updated and locally validated; deployment and runtime
verification remain operator steps. Two additional spare cores on CPU30/31
expand the RT pool to 34 physical cores; further growth is deferred.

## Objective and scope

Document the RTC's current thread and IRQ assignments, replace misleading
`spareXX` names, and design an allocation with room for development and future
growth. First complete the deployed inventory below; then agree on names,
sharing and CPU capacity before implementing changes.

Identify every assignment by **host, cpuset, CPU and thread**. RTC and ICC have
independent cpuset namespaces. Matching names or CPU numbers on the two hosts
do not constitute a conflict.

All instrument commands and configuration changes are performed by the
operator. Agents work from workstation copies of files and operator reports.

## Branch and commit scope

The RTC scripts, cleanup tests and this engineering record belong to
`jrmales/rtc-cpuset-restructure`, based directly on `dev` at `cd37915d`.
That base does not contain dmSyncopation. Its source/example cpuset change
is preserved separately on `jrmales/dm-syncopation` in `6e3d454e`, followed
by the corresponding current documentation in `c4ac2272`. The explicit local
synco config selects CPU41 independently of that source-default change.

The separate config repository contains the 17 matching config changes in
`14401b2` on `jrmales/rtc-cpuset-restructure`, based on its `onsky` branch;
that repository has no `dev` branch. The migration requires the matching
scripts and configs together.

## Sources and meaning of the baseline

The baseline combines:

- [RTC make_cpusets](../../../rtSetup/RTC/make_cpusets).
- Local `/opt/MagAOX/config/proclist_RTC.txt` and each available
  `/opt/MagAOX/config/<process-name>.conf`.
- Local `cacao/woofer-vispyr/fpssetup.setval.woofer-vispyr.conf` and
  `cacao/tweeter-vispyr/fpssetup.setval.tweeter-vispyr.conf`, under that config
  directory, with their corresponding `cacaovars.*.bash` substitutions.
- [RTC move_irqs](../../../rtSetup/RTC/move_irqs),
  [irqbalance settings](../../../rtSetup/RTC/irqbalance) and
  [device policy](../../../rtSetup/RTC/irqbalance_policyscript).
- [ICC process list](/opt/MagAOX/config/proclist_ICC.txt) and
  [ICC make_cpusets](../../../rtSetup/ICC/make_cpusets), solely to establish
  ownership of configurations shared between host inventories.
- Source defaults from `jrmales/dm-syncopation` at `90f452ce`, and the recent
  operator-reported timing experiments where the local configuration snapshot
  is incomplete.
- Operator-provided RTC `lscpu -e` and CPU33/CPU61 L3 sysfs output, received
  2026-10-26 in attachment
  `ed8ba63e-06f8-4610-b8fe-6d55438ed39a/Pasted text.txt`.

The baseline below is the **pre-migration** configured inventory, not a current
runtime affinity measurement. The migration map and execution notes record
the subsequent workstation edits. Source defaults in the baseline refer to
the version before this migration, not the updated synco default below.
An uncommented process-list entry does not establish that the process is
running. A missing explicit assignment does not establish that a CPU is free.

The refreshed local RTC process list has **89 uncommented entries**. Exact
configuration files were found for **82**. Of these, **27 processes have 32
explicit cpuset settings**, all naming sets defined in RTC's script; **55**
specify no cpuset. The seven missing exact configurations are recorded below
for reconciliation. `dmtweeter-synco` is in the group without explicit cpuset
settings: its primary-monitor placement comes from the source default. Including
that default gives **28 assigned processes / 33 placement roles**, separately
from the explicit-config count. These counts describe the synced workstation
snapshot, not a runtime census.

## Pre-migration RTC CPU map

The pre-migration script places the background pool on CPUs `0-31,64-95`,
reserves CPUs `32-63` as 32 single-thread real-time cores, and offlines CPUs
`96-127`.
It assumes a 64-core/128-thread machine with those sibling relationships.
Each named set on CPUs `32-63` has `cpu_exclusive=1` and `mems=0`.

The operator identifies the RTC processor as **AMD Ryzen Threadripper PRO
5995WX, 64 cores**. It has eight Zen 3 core chiplets (CCDs), each containing
eight cores sharing a 32 MB L3 cache. The L3 total is distributed across these
domains; one socket does not imply one shared last-level cache. See
[AMD's architecture presentation, slide 23](https://gpuopen.com/download/GDC_AMD_Ryzen_Processor_Software_Optimization.pdf#page=23).
The operator's topology readback confirms the following Linux L3-sharing
groups, consistent with those eight CCDs. These are observed cache IDs, not
an assumption based on consecutive CPU numbers.

### Operator-confirmed topology

| Linux L3 ID / cache domain | Physical core IDs / primary CPUs | Other online SMT CPUs | Pool in pre-migration setup script |
| --- | --- | --- | --- |
| 0 | 0-7 | 64-71 | `system` |
| 1 | 8-15 | 72-79 | `system` |
| 2 | 16-23 | 80-87 | `system` |
| 3 | 24-31 | 88-95 | `system` |
| 4 | 32-39 | None | Real-time |
| 5 | 40-47 | None | Real-time |
| 6 | 48-55 | None | Real-time |
| 7 | 56-63 | None | Real-time |

All online CPUs report **socket 0 / NUMA node 0**. There are **96 online
logical CPUs spanning 64 physical cores**. Cores0-31 each have two online SMT
threads; cores32-63 each have one. CPUs96-127 are offline, matching the setup
script's mask. Their `CORE`, `SOCKET` and `NODE` fields are `-` in this
readback, so it does not independently establish every offline sibling pair.

The direct sysfs readback identifies `cpu33/cache/index3/level=3` with
`shared_cpu_list=32-39`, and CPU61's corresponding list as `56-63`.
Thus CPU33 and CPU61 are on different physical cores and different L3 domains,
while CPU33 and CPU34 are different physical cores sharing L3. Each of these
three cores has only one online SMT thread. One exposed NUMA node does not
imply one L3 domain or establish PCIe/DMA memory locality. This readback does
not measure current thread/IRQ affinities, competing workload or communication
latency.

### Configured assignments

Application names below map to their `<process-name>.conf` files. **FG** means
the framegrabber/acquisition thread; **SM** means a shared-memory monitor.
CACAO names identify processes rather than MagAOX app instances. IRQ entries
describe the startup policy; later manual differences are kept separate.

| CPU | Existing cpuset | Configured users / purpose | Operator correction or confirmation |
| --- | --- | --- | --- |
| 0-31, 64-95 | `system` | Background tasks and ordinary IRQs | |
| 32 | `camwfs_sw` | `camwfs-sw` FG | |
| 33 | `camwfs` | `camwfs` FG; EDT IRQ under startup policy | |
| 34 | `camwfs2` | No assignment found in reviewed RTC sources | use for IRQ, rename to camwfs_irq|
| 35 | `spare35` | `aol1_modevalWFS-sw` FG | |
| 36 | `spare36` | `dmtweeter-recon` DM-command SM and FG | |
| 37 | `dm00comb` | CACAO `DMch2disp-00` | |
| 38 | `woofer` | `dmwoofer` command SM; `t2wOffloader` SM; ALPAO IRQ | |
| 39 | `dm01comb` | CACAO `DMch2disp-01` | |
| 40 | `tweeter` | `dmtweeter` command SM | |
| 41 | `t2w` | No assignment found; `t2wOffloader` currently requests `woofer` | Corrected: rename `dmtweeter-synco` and move synco here |
| 42 | `spare42` | `aol0_modevalDMf_mon-sw` FG | rename |
| 43 | `spare43` | `po4ao` SM / inference thread, requested FIFO45 | Config synced by operator, rename |
| 44 | `spare44` | `aol1_modevalDMf_mon-sw` FG | rename |
| 45 | `acqu_mvm` | CACAO `acquWFS-{0,1}`, `wfs2cmodeval-{0,1}`, `mvalC2dm-{0,1}`; `loPredCtrl` SM and FG; `nnReconstructor` FG; `dlDataCollection` SM; NVIDIA IRQs | NN FG and data-collection SM each request FIFO45 in synced configs |
| 46 | `mfilt` | CACAO `mfilt-0`, `mfilt-1` | |
| 47 | `tweeterSpeck` | `tweeterSpeck` and `tweeterMod` modulator threads | |
| 48 | `mlat` | CACAO `mlat-0`, `mlat-1` | |
| 49 | `RTmon` | No assignment found in reviewed RTC sources | confirmed, this is used by milk-shmimmon|
| 50 | `spare50` | `aol1_modevalWFS_cbuff` SM and FG | rename |
| 51 | `spare51` | `dm01disp-sw` FG | rename |
| 52 | `spare52` | `dm01disp03-sw` FG | rename |
| 53 | `spare53` | `aol1_imWFS2-sw` FG | rename |
| 54 | `camtip` | `camtip` FG | Corrected: retain name and placement |
| 55 | `camtip_sw` | `camtip-sw` FG | Corrected: retain name and placement |
| 56 | `camacq` | `camacq` FG | rename spare56, will drop from camacq |
| 57 | `camacq_sw` | `camacq-sw` FG | rename spare 57, will drop from camacq |
| 58 | `shmimTCP` | Network IRQ matching `enx98b78501aebc`; no listed application explicitly assigned | confirmed |
| 59 | `spare59` | `acc000-cbuff` SM and FG | rename |
| 60 | `spare60` | `dm00disp-sw` FG | rename |
| 61 | `spare61` | `nnReconstructor` SM / inference thread, requested FIFO45; `dmtweeter-synco` primary SM, default FIFO30 | Both listed; synco config retains source placement default. Current EDT IRQ placement still needs confirmation. Use for nnReconstructor, rename |
| 62 | `spare62` | `dm00disp03-sw` FG | rename |
| 63 | `spare63` | `acc000-sw` FG | rename |
| 96-127 | None | Offlined SMT siblings under the current script | |

Fourteen `spareXX` sets already have explicit users in this RTC baseline:
`35,36,42,43,44,50,51,52,53,59,60,61,62,63`. Synco's source default adds another
user of the already occupied CPU61. CPUs34,41,49 had no found assignment in
the initial source/config map. The operator subsequently assigned EDT to
CPU34, synco to CPU41 and confirmed milk-shmimmon on CPU49; none is classified
as reserve capacity in the migration map.

### Application-thread interpretation

- `streamWriter` consumes `[framegrabber].cpuset`, pinning its acquisition
  thread. Its separate `[writer].cpuset` is unspecified in these configs;
  `camtip-sw` has only a commented writer assignment.
- `dmtweeter-recon` uses `[dmCommand].cpuset` and `[framegrabber].cpuset`.
  The circular-buffer apps use `[shmimMonitor].cpuset` and
  `[framegrabber].cpuset`. Record both threads when changing their placement.
- `[dm].cpuset` assigns the controller's command-monitor thread.
- `mzmqServerRTC`, `mzmqServerRTCcamwfs`, `mzmqServerRTC_calib` and
  `sysMonRTC` have no explicit cpuset in the reviewed configuration/source.
  Their names do not assign them to `shmimTCP` or `RTmon`.
- An empty cpuset means inherited placement, ordinarily the `system` pool
  under the startup policy. It is not an explicit whole-process affinity.
  Record helper/CUDA threads separately when verifying runtime placement.
- CPU exclusivity is between cpusets; several threads can share one set.
  The shared CPU38, CPU45 and CPU47 placements above are explicit. CPU61 now
  also has configured sharing between NN inference (FIFO45) and synco (FIFO30).
  The EDT IRQ was on CPU61 in recent captures; its present placement is still
  unverified, so do not count it as a freshly confirmed third user.
- `nnReconstructor`'s `[shmimMonitor]` reads `aol1_imWFS2` and runs the
  preprocessing/inference callback. Its `[framegrabber]` publishes
  `aol1_modevalWFS` on a separate thread. `po4ao`'s monitor consumes
  `aol1_modevalWFS`. The two NN cpuset entries do not establish placement of
  CUDA/TensorRT helper threads.

### CACAO startup placement

Both RTC FPS setup files specify a cpuset and a numeric `taskset`; update both
when moving a process. Loop 0/DM00 is woofer and loop 1/DM01 is tweeter.

| Processes | Cpuset | CPU | Configured RT priority |
| --- | --- | --- | --- |
| `DMch2disp-00` | `dm00comb` | 37 | 49 |
| `DMch2disp-01` | `dm01comb` | 39 | 49 |
| `acquWFS-0`, `acquWFS-1` | `acqu_mvm` | 45 | 49 |
| `wfs2cmodeval-0`, `wfs2cmodeval-1` | `acqu_mvm` | 45 | 48 |
| `mvalC2dm-0`, `mvalC2dm-1` | `acqu_mvm` | 45 | 48 |
| `mfilt-0`, `mfilt-1` | `mfilt` | 46 | 48 |
| `mlat-0`, `mlat-1` | `mlat` | 48 | Not specified in these files |

The `rtc_procset` helper mentioned in `make_cpusets` was not found in the
reviewed repository/local installation. Confirm whether the RTC has another
placement helper before treating this source inventory as complete.

### IRQ placement

| IRQ match | Startup CPU(s) | Known later difference / operator update |
| --- | --- | --- |
| Ordinary IRQs | `0-31,64-95` | |
| `edt` | 33, from `camwfs` | Operator moved IRQ405 to CPU61 during recent camera measurements; confirm present placement and desired persistent policy |
| `nvidia` | 45, from `acqu_mvm` | |
| `cp2x72c` / ALPAO | 38, from `woofer` | |
| `enx98b78501aebc` / shmimTCP NIC | 58, from `shmimTCP` | |

The camera timing record,
`agents/plans/2026-10/ocam2k-acquisition-publication-timing.md` on
`jrmales/dm-syncopation`, records the EDT change and operator confirmation.
This is historical evidence,
not a fresh runtime readback. The pre-migration irqbalance settings exclude
`32-63,96-127` from ordinary balancing, with the device-pattern policy protecting
the selected IRQs from redistribution.

## ICC ownership: keep separate from RTC cleanup

The following are ICC assignments, confirmed by `proclist_ICC.txt` and ICC's
own cpuset script. They do not occupy RTC CPU capacity.

| Process | Host | Cpuset | Host CPU | Independent RTC use of that CPU number |
| --- | --- | --- | --- | --- |
| `acc001-cbuff` | ICC | `spare61` | 61 | NN inference and synco; recent manual EDT placement still requires confirmation |
| `aol2_modevalWFS-sw` | ICC | `spare62` | 62 | `dm00disp03-sw` |
| `aol2_modevalWFS_cbuff` | ICC | `spare63` | 63 | `acc000-sw` |

The `aol2_*` processes belong to ICC. Validate their configurations against
`rtSetup/ICC/make_cpusets`, not RTC's definitions. Keep future RTC renames scoped
to RTC consumers even where ICC uses the same old cpuset name.

## Operator additions: deployed and uncommitted RTC differences

Edit this section before settling a new allocation. Include processes launched
manually or outside `proclist_RTC.txt`, alternate config files, runtime FPS
changes, taskset changes and IRQ changes. Record the actual thread role rather
than only its process name. Add rows as needed; unknown cells deliberately
remain TBD.

| Process / thread or IRQ | Synced configuration / reason to reconcile | Configured cpuset | Configured CPU(s) | Requested policy / priority | Config source; runtime confirmation / operator notes |
| --- | --- | --- | --- | --- | --- |
| `dmtweeter-synco` primary monitor | Newly listed; config names `dm01disp03` and omits cpuset/priority overrides | `spare61` (source default) | 61 | FIFO30 (source default) | `/opt/MagAOX/config/dmtweeter-synco.conf`; runtime confirmation pending |
| `dmtweeter-synco` camera/combiner observers | No overrides; default empty cpuset | Inherited | Inherited | SCHED_OTHER / priority0 | Source defaults; runtime confirmation pending |
| `nnReconstructor` input SM / inference | Synced `[shmimMonitor]` consumes `aol1_imWFS2` | `spare61` | 61 | FIFO45 | `/opt/MagAOX/config/nnReconstructor.conf:4-5`; runtime confirmation pending |
| `nnReconstructor` FG / publisher | Synced `[framegrabber]` publishes `aol1_modevalWFS` | `acqu_mvm` | 45 | FIFO45 | `/opt/MagAOX/config/nnReconstructor.conf:10-11`; runtime confirmation pending |
| `po4ao` SM | Synced `[shmimMonitor]` consumes `aol1_modevalWFS` | `spare43` | 43 | FIFO45 | `/opt/MagAOX/config/po4ao.conf:4-5`; runtime confirmation pending |
| `dlDataCollection` SM | Newly listed; synced `[shmimMonitor]` assignment | `acqu_mvm` | 45 | FIFO45 | `/opt/MagAOX/config/dlDataCollection.conf:4-5`; runtime confirmation pending |
| EDT framegrabber IRQ | Startup CPU33; recent measurements used CPU61 | N/A | TBD | N/A | |
| `camtipSR` development threads | RTC process-list entry; exact local `.conf` absent | TBD | TBD | TBD | |
| `windsocc` / `windsoccRT` threads | RTC process-list entry; exact local `.conf` absent | TBD | TBD | TBD | |
| `loPredCtrlAcc` threads | RTC process-list entry; exact local `.conf` absent | TBD | TBD | TBD | |
| `dmTemporalResponse` threads | RTC process-list entry; exact local `.conf` absent | TBD | TBD | TBD | |
| Additional process / thread | | TBD | TBD | TBD | |
| Additional process / thread | | TBD | TBD | TBD | |

The remaining seven missing exact configurations are the four app instances
`camtipSR`, `windsocc`, `loPredCtrlAcc`, `dmTemporalResponse` and the three
tunnel instances below. No previously recorded explicit cpuset assignment
changed or disappeared in this sync; the explicit additions are NN's two
threads, PO4AO's monitor and the data-collection monitor. `dmtweeter-synco`
and `dlDataCollection` are the two new process-list entries.

Also reconcile the missing exact `rtc_icc_indi.conf`, `rtc_aoc_indi.conf` and
`rtc_acc000_indi.conf` files. These are `sshDigger` instances; determine their
actual configuration mechanism rather than assuming a missing MagAOX-style
file implies that the instance cannot run.

Two legacy RTC-looking configs need a separate status decision:

| Config / instance | Referenced cpuset | Current evidence | Decision |
| --- | --- | --- | --- |
| `aol1_modevalDM-sw` | `spare34`, undefined in RTC script | Listed in `isRTC.conf` driver roster, absent from every reviewed `proclist*.txt` | TBD: retain/update or retire |
| `aol1_modevalDM_cbuff` | `spare15`, undefined in RTC script | Listed in `isRTC.conf` driver roster, absent from every reviewed `proclist*.txt` | TBD: retain/update or retire |

Reconcile the process list and `isRTC.conf` roster when adding or retiring apps.
Other listed apps with no explicit cpuset may also have uncommitted RTC changes;
the editable rows above are a starting point, not an exhaustive runtime census.

## Restructuring decisions

Separate **renaming**, **moving/sharing workloads**, and **adding CPU capacity**.
A rename alone preserves the CPU and priority. Adding named sets within the
existing pool does not add physical cores. More real-time physical cores would
require a deliberate change to the current system/real-time partition.

| Decision | Current baseline | Proposed decision / operator answer |
| --- | --- | --- |
| Naming convention | Mixture of device, pipeline and `spareXX` names | Use owning app names for occupied spare sets; retain unmarked existing names |
| Unit represented by a cpuset | Some dedicated threads; some shared pipeline groups | Retain existing grouping except separate EDT and synco placements |
| Existing users of `spareXX` | Fourteen occupied RTC sets | Rename all fourteen to owning app names |
| Controller/offload sharing | `t2wOffloader` shares `woofer`; `t2w` on CPU41 has no found user | Keep offloader sharing; repurpose CPU41 for synco |
| Acquisition/reconstruction sharing | Both RTC loop paths, `loPredCtrl`, NN publishing and `dlDataCollection` use CPU45 | Keep current sharing |
| NN inference / syncopation sharing | NN input/inference thread requests FIFO45 on CPU61; synco primary thread defaults to FIFO30 there | NN stays on CPU61; synco moves to dedicated CPU41 at FIFO30 |
| Modulation sharing | `tweeterSpeck` and `tweeterMod` use CPU47 | Keep current sharing |
| EDT versus synco placement | Persistent and recent manual configurations differ | EDT CPU34 (`camwfs_irq`); synco CPU41 (`dmtweeter-synco`) |
| Previously unassigned reserved cores | CPUs34,41,49 lacked found assignments | EDT on CPU34; synco on CPU41; operator confirms milk-shmimmon on CPU49 |
| Development reserve | Camacq had dedicated CPU56/57 sets | Return camacq threads to ordinary scheduling; reserve `spare30`, `spare31`, `spare56` and `spare57` |
| Growth within original real-time pool | 32 physical cores reserved before migration | Retain the original 32 dedicated cores |
| Growth beyond original pool | System owns cores represented by CPUs0-31 and their siblings64-95 | Transfer physical cores30/31 to dedicated spares; system becomes `0-29,64-93` |
| SMT policy | CPUs96-127 confirmed offline; RT cores32-63 each have one online thread; system cores0-31 each have two | Offline94-127 so new RT cores30/31 also have no online SMT sibling |
| NUMA/locality policy | All online CPUs report node0; current cpusets use `mems=0`; eight L3 domains confirmed | Retain memory masks; EDT/FG on separate cores within L3 domain4 |

The pre-migration partition assigns four complete L3 domains to `system` and
four to real-time work. The operator selected a two-core expansion: physical
cores30/31 become dedicated reserves, their SMT siblings94/95 are offlined,
and the system loses those four logical CPUs. The new reserves share L3
domain3 with system cores24-29. This deliberately trades complete cache-domain
separation for adding two cores without transferring an entire eight-core
domain. Revalidate topology if hardware or BIOS settings change.

### Placement locality and the camera pair

The selected map puts the EDT IRQ on CPU34 and `camwfs` FG on CPU33, separate
physical cores sharing L3. Separate cores permit concurrent execution;
shared-cache locality can reduce handoff and data-sharing cost. SMT siblings
still share physical execution resources and do not provide that same
separation. Include memory/PCIe locality when
interpreting the operator's NUMA/topology information.

The earlier IRQ-placement comparison in that camera timing record moved
only the EDT IRQ from CPU33 to CPU61 while FG remained on CPU33. Across
the retained summary windows, publication interval SD fell from 1.864 to
0.995 us, while mean acquisition-to-write latency increased by 0.856 us.
Traces showed the previous same-core hard IRQs interrupting acquisition work.
That comparison predates deferred rearm and reduced housekeeping, and does
not isolate cache-handoff costs or establish CPU61 as the best current choice.

The new topology readback establishes that this experiment moved the IRQ
from L3 domain4 to domain7, as well as off the FG core. It therefore does not
compare separate cores within one CCD against separate CCDs. The improvement
despite crossing L3 domains supports prioritizing removal of same-core IRQ
interference, while leaving the best separate-core placement to measurement.
The driver also has shared-lock/serial handling that CPU separation cannot
eliminate. CPU61's pre-migration NN/synco sharing is removed by moving synco
to CPU41; account for actual competing activity in any new comparison.

The operator confirms CPU34's former `camwfs2` set was a placeholder for
possible future descramble/FG processing. It is repurposed as `camwfs_irq`,
a separate physical core in domain4 with no online SMT sibling. Domain4 also
contains the camera writer, reconstruction and DM-combiner/woofer workloads
listed above; shared L3 is not an otherwise empty resource. Verify the new
EDT IRQ affinity after operator deployment and qualify timing against the
previous captures.

Record the following for each proposed latency-sensitive placement before
comparing timing: physical core, online SMT siblings, shared-L3 group, NUMA
node, memory policy, device/IRQ affinity and competing threads. Linux exposes
cache-sharing groups through `cpu*/cache/index*/shared_cpu_list`; verify the
corresponding `level` attribute when identifying L3. See the
[kernel CPU/cache ABI](https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/master/Documentation/ABI/testing/sysfs-devices-system-cpu).

### First migration assignment map

Keep existing app priorities and the original CPU32-63 dedicated assignments.
Expand the partition with two new spare cores on CPU30/31. Names for occupied
spare sets follow the owning app instance. The operator selected
CPU34 for the EDT IRQ, CPU41 for `dmtweeter-synco`, CPU61 for NN inference, and
confirmed CPU49 belongs to `milk-shmimmon`. `camwfs2` was a placeholder for
possible future camera processing, so no existing processing thread needs
relocation from CPU34. The camera IRQ now uses that separate core in the
same L3 domain as FG on CPU33.

The target partition is:

| Role | Logical CPUs | Physical cores / online threads |
| --- | --- | --- |
| System tasks and ordinary IRQs | `0-29,64-93` | 30 / 60 |
| Dedicated real-time sets | `30-63` | 34 / 34 |
| Offline SMT siblings | `94-127` | 34 offline logical CPUs |

This gives 94 online logical CPUs. The 96-thread topology readback above is
the measured pre-migration baseline. IRQ balancing excludes
`30-63,94-127`; device-specific IRQ routing remains on the assigned RT cores.

| CPU | Old cpuset | New cpuset | Migration behavior |
| --- | --- | --- | --- |
| 30 | Part of `system` | `spare30` | Dedicated reserve; offline SMT sibling94 |
| 31 | Part of `system` | `spare31` | Dedicated reserve; offline SMT sibling95 |
| 32 | `camwfs_sw` | Unchanged | `camwfs-sw` FG |
| 33 | `camwfs` | Unchanged | `camwfs` FG; EDT IRQ moves to CPU34 |
| 34 | `camwfs2` | `camwfs_irq` | Dedicated EDT IRQ |
| 35 | `spare35` | `aol1_modevalWFS-sw` | Rename only |
| 36 | `spare36` | `dmtweeter-recon` | Rename both DM-command and FG assignments |
| 37 | `dm00comb` | Unchanged | CACAO woofer combiner |
| 38 | `woofer` | Unchanged | Keep controller/offloader/ALPAO IRQ sharing |
| 39 | `dm01comb` | Unchanged | CACAO tweeter combiner |
| 40 | `tweeter` | Unchanged | `dmtweeter` controller |
| 41 | `t2w` | `dmtweeter-synco` | Move primary monitor/scheduler from CPU61; retain FIFO30 |
| 42 | `spare42` | `aol0_modevalDMf_mon-sw` | Rename only |
| 43 | `spare43` | `po4ao` | Rename only |
| 44 | `spare44` | `aol1_modevalDMf_mon-sw` | Rename only |
| 45 | `acqu_mvm` | Unchanged | Keep existing CACAO/NN publisher/data collection/GPU IRQ sharing |
| 46 | `mfilt` | Unchanged | CACAO modal filters |
| 47 | `tweeterSpeck` | Unchanged | Modulator sharing |
| 48 | `mlat` | Unchanged | CACAO latency measurement |
| 49 | `RTmon` | Unchanged | Operator confirms `milk-shmimmon` |
| 50 | `spare50` | `aol1_modevalWFS_cbuff` | Rename both SM and FG assignments |
| 51 | `spare51` | `dm01disp-sw` | Rename only; existing combined-command recorder |
| 52 | `spare52` | `dm01disp03-sw` | Rename only |
| 53 | `spare53` | `aol1_imWFS2-sw` | Rename only |
| 54 | `camtip` | Unchanged | Operator corrected initial rename annotation |
| 55 | `camtip_sw` | Unchanged | Operator corrected initial rename annotation |
| 56 | `camacq` | `spare56` | Remove camacq assignment; retain reserve |
| 57 | `camacq_sw` | `spare57` | Remove camacq writer assignment; retain reserve |
| 58 | `shmimTCP` | Unchanged | Dedicated NIC IRQ |
| 59 | `spare59` | `acc000-cbuff` | Rename both SM and FG assignments |
| 60 | `spare60` | `dm00disp-sw` | Rename only |
| 61 | `spare61` | `nnReconstructor` | NN inference stays here at FIFO45; synco moves to CPU41 |
| 62 | `spare62` | `dm00disp03-sw` | Rename only |
| 63 | `spare63` | `acc000-sw` | Rename only |

`camacq.conf` and `camacq-sw.conf` drop their FG cpuset keys. Both set
`framegrabber.threadPrio=0`, and `camacq-sw` also sets `writer.threadPrio=0`,
selecting `SCHED_OTHER` instead of the helpers' positive FIFO defaults. Their
threads inherit the ordinary system pool. The old dedicated CPU56/57 slots
remain within the RT partition, available for the target allocation above.

The local `dmtweeter-synco.conf` explicitly sets
`shmimMonitor.cpuset=dmtweeter-synco`. The companion feature branch's source
default and example use the same name, so both configured and default startup
use CPU41. No `dmtweeter-sw` instance or extra recorder is added.

The final local process-list audit finds **26 configs / 31 explicit cpuset
settings**, all resolving to the new RTC map. The two camacq configs now
have ordinary scheduling and no cpuset key; synco gains an explicit key. The
seven missing exact configs listed in the baseline remain unchanged.

Active CACAO cset/taskset pairs stay unchanged. Ordinary IRQ affinity changes
to `0-29,64-93`, and the irqbalance exclusion list changes to `30-63,94-127`.
ICC's independent `spare61`/`spare62`/`spare63` consumers are excluded from this
migration. The two undefined legacy RTC-looking configs remain a separate
retain/retire decision.

### Operator removal and recreation procedure

The [remove_cpusets helper](../../../rtSetup/RTC/remove_cpusets) defaults to
dry run. It preserves the root and `system`, verifies the configured cgroup-v1
cpuset mount, and accepts only known legacy/current RTC top-level names. It
checks all target groups and descendants for tasks before removing any group,
then removes empty children before their parents using `rmdir`. Occupied or
unknown sets stop the operation with a report; the helper never kills threads
or migrates tasks.

After deploying the matching workstation configs/scripts to RTC, the operator
performs this sequence from the MagAOX source checkout:

1. Stop the AO/CACAO processes and MagAOX apps, including manually launched
   monitors such as `milk-shmimmon`. Prevent automatic restarts during the
   migration. Retain the old scripts/configs for rollback.
2. Preview and remove the empty groups:

   ```bash
   sudo python3 rtSetup/RTC/remove_cpusets --dry-run
   sudo python3 rtSetup/RTC/remove_cpusets --apply
   ```

   If threads are reported, stop those owners and retry. A concurrent restart
   can interrupt removal after some groups were removed; keep processes
   stopped and rerun the helper. Do not rebuild while occupied old exclusive
   sets remain.
3. Recreate the groups, install the matching IRQ balancing policy and apply
   device IRQ routing:

   ```bash
   sudo bash rtSetup/RTC/make_cpusets
   sudo install -m 0644 rtSetup/RTC/irqbalance /etc/sysconfig/irqbalance
   sudo bash rtSetup/RTC/move_irqs
   ```

4. Restart apps/CACAO with the matching configs. Verify actual thread
   affinities/priorities and EDT IRQ effective affinity on CPU34, then take
   a timing comparison. Confirm system CPUs `0-29,64-93`, dedicated cores
   `30-63` and offline siblings `94-127`. Running `make_cpusets` alone does
   not move IRQs or install the irqbalance config.

`make rtscripts_install MAGAOX_ROLE=RTC` also installs `remove_cpusets` beside
`make_cpusets` and `move_irqs`; the installed helper finds its sibling setup
script to recognize current names. It does not automatically execute cleanup.

### Changed files and local validation

Files on the dev-based cpuset branch:

- `rtSetup/RTC/make_cpusets`, `rtSetup/RTC/move_irqs`, `rtSetup/RTC/irqbalance`,
  `rtSetup/RTC/remove_cpusets` and `rtSetup/RTC/tests/remove_cpusets_test.py`.
- `Makefile` (`rtscripts_install` includes the cleanup helper).
- This plan. Historical capture records retain their original CPU/set names
  on their original feature branch.

Companion files committed on `jrmales/dm-syncopation`:

- `apps/dmSyncopation/dmSyncopation.hpp`,
  `apps/dmSyncopation/dmtweeter_synco.conf`,
  `apps/dmSyncopation/doc/dmSyncopation.md` and
  `apps/dmSyncopation/doc/camera-acquWFS-test.md`.

The separate `/opt/MagAOX/config` commit contains **17 changed files**:

- `dmtweeter-recon.conf`, `dmtweeter-synco.conf`, `nnReconstructor.conf`
  and `po4ao.conf`.
- `aol0_modevalDMf_mon-sw.conf`, `aol1_modevalDMf_mon-sw.conf`,
  `aol1_modevalWFS-sw.conf`, `aol1_modevalWFS_cbuff.conf` and
  `aol1_imWFS2-sw.conf`.
- `dm00disp-sw.conf`, `dm00disp03-sw.conf`, `dm01disp-sw.conf` and
  `dm01disp03-sw.conf`.
- `acc000-cbuff.conf`, `acc000-sw.conf`, `camacq.conf` and `camacq-sw.conf`.

Local verification covers all 34 distinct RT core assignments and all explicit
RTC config cpuset references, preserved ICC/CACAO files, camacq's zero
priorities, synco's matching source/example/local-config name, shell syntax,
consistent system/offline/IRQ masks, companion header clang-format, Markdown
links and whitespace. All ten cleanup tests pass using temporary trees and
mocked removals. No actual cgroup, IRQ or instrument operation was executed.
Deployment, runtime thread/IRQ readback
and the post-migration timing comparison remain operator follow-up steps.

## Implementation dependencies and sequence

1. **Complete the operator inventory.** Record deployed/uncommitted configs,
   runtime overrides, thread placement, active IRQ placement and topology.
   Resolve whether apparently unassigned sets have development users.
2. **Agree on the target map.** Settle naming, shared versus dedicated cores,
   system capacity and development reserve. Review CPU moves independently of
   name changes; retain rollback copies of the deployed map/configs.
3. **Prepare matching source/config changes.** Update RTC `make_cpusets`, all
   affected app configs, CACAO `.procinfo.cset` and `.procinfo.taskset`, IRQ
   routing and irqbalance exclusions/policy. Update source defaults, examples
   and current procedures that name moved sets. On the companion feature
   branch, `apps/dmSyncopation/dmSyncopation.hpp` and
   `apps/dmSyncopation/dmtweeter_synco.conf` now name `dmtweeter-synco` on CPU41.
   Preserve historical capture values in engineering records.
4. **Validate the proposed configuration locally.** Check that every RTC
   reference resolves, host ownership is correct, exclusive sibling sets do
   not overlap, all CPUs/memory nodes exist, CACAO cset/taskset agree and
   system/IRQ/SMT masks match the chosen partition. Validate touched scripts
   without executing their hardware operations.
5. **Plan the operator migration.** `make_cpusets` uses `mkdir -p` and does not
   remove old sets. A renamed exclusive set can conflict with its predecessor
   on the same CPUs. Specify how affected workloads are quiesced and obsolete
   sets drained/removed; simply rerunning the script is not a complete rename.
6. **Deploy together and verify on RTC.** The operator installs the prepared
   scripts/configs, restarts affected workloads and reports actual per-thread
   cpuset/affinity, scheduling policy/priority, IRQ placement and online CPUs.
   Check helper/CUDA threads as well as main threads. Record persistent startup
   behavior and a short loop/camera/DM timing comparison against the baseline.
7. **Close the record.** Replace TBD entries with the final map, record
   deployment/version information and reconcile the config/process rosters.

The repository's `rtscripts_install` target selects RTC/ICC scripts through
`MAGAOX_ROLE` and installs `make_cpusets`, `move_irqs` and `remove_cpusets`
(a no-op stub for roles without a removal helper). It does **not** deploy
the irqbalance settings or policy script; include those separately when the
chosen partition or device routing changes. Keep ICC installation/configuration
independent of the RTC rollout.

## Review and execution log

- 2026-10-26: Created workstation inventory and editable development/capacity
  planning sections. ICC-owned `aol2_*` and `acc001-cbuff` assignments explicitly
  separated from RTC occupancy. No cpuset/configuration changes implemented.
- 2026-10-26: Operator synced updated configs. Reparsed all RTC entries:
  89 listed processes, 82 exact configs, 27 processes / 32 explicit cpuset keys,
  55 configs without an explicit cpuset and seven missing exact configs.
  Added PO4AO on CPU43, NN inference on CPU61 and NN publishing on CPU45,
  synco's source-default CPU61 placement, and newly listed `dlDataCollection`
  on CPU45. CPU61 sharing is now a configured/default assignment; current IRQ
  placement and actual thread affinities remain to be confirmed. ICC ownership
  remains separate. No cpuset rename or relocation implemented.
- 2026-10-26: Operator identifies RTC CPU as Threadripper PRO 5995WX. Added
  cache-domain/topology requirements and recorded the earlier separate-core
  EDT/FG result as a starting point for placement experiments. Exact current
  core/cache/NUMA mapping and the target allocation remain open.
- 2026-10-26: Operator supplied RTC topology readback: 96 online threads on
  64 cores, all reporting socket0/node0, eight L3 groups, and CPUs96-127
  offline. Recorded the full cache-domain map and current whole-domain
  system/RT split. Confirmed CPU33/CPU61 cross L3 domains; CPU34 shares L3
  with CPU33 on a separate physical core and is a conditional test candidate.
  Current IRQ placement, runtime occupancy and the target allocation remain
  open. No instrument settings changed.
- 2026-10-26: Operator annotated the baseline and authorized the first rename
  migration. Updated RTC `make_cpusets`, routed EDT through `camwfs_irq` on
  CPU34, and updated 16 unambiguous local configs, including ordinary
  scheduling for camacq camera/writer threads. Added `remove_cpusets` and its
  install target plus ten mocked-tree tests. Static validation confirms 32
  distinct RT CPU assignments and all edited config names resolve; shell
  syntax, cleanup tests and whitespace checks pass. ICC and active CACAO
  placements remain unchanged. CPU54/55 names, synco's replacement placement
  and the new CPU41 recorder's stream remain pending operator clarification.
  No hardware operations or commits performed.
- 2026-10-26: Operator corrected CPU41 to `dmtweeter-synco` and retained
  `camtip`/`camtip_sw` on CPU54/55. Finalized CPU56/57 as `spare56`/`spare57`,
  moved synco's default/example/local-config placement to CPU41, and retained
  NN inference on CPU61. Seventeen local configs are modified. Final audit:
  32 unique RT CPUs, 26 configs / 31 explicit cpuset keys all resolving to
  the new map, ten cleanup tests passing, and syntax/format/docs checks clean.
  No extra recorder added; no instrument operations or commits performed.
- 2026-10-26: Operator requested two more reserves on physical cores30/31.
  Added `spare30`/`spare31`, reduced `system` and ordinary IRQ affinity to
  `0-29,64-93`, and expanded offline siblings to `94-127` and irqbalance
  exclusions to `30-63,94-127`. Target: 34 RT cores, 30 system cores with
  60 online threads, 94 online logical CPUs total. Documented shared L3
  domain3 with system cores24-29 and the explicit irqbalance config install
  step. Existing app placements, local app configs and ICC remain unchanged.
  Local checks confirm all 34 exclusive core assignments, matching masks,
  all 31 config references and cleanup recognition of the two new names;
  ten mocked cleanup tests and shell syntax/whitespace checks pass.
- 2026-10-26: Separated the standalone migration onto
  `jrmales/rtc-cpuset-restructure`, directly from refreshed `dev` at `cd37915d`.
  The seven-file change includes only the RTC scripts, installer hook, cleanup
  tests and this record. Synco-specific integration remains on its original
  feature branch in `6e3d454e`/`c4ac2272`; matching config changes are committed
  separately in `14401b2`. No instrument deployment or push performed.
- Operator additions: Remaining missing/legacy configs are outside this migration.
- Target allocation review: First migration approved through operator annotations
  and CPU41/54/55 corrections, plus the CPU30/31 expansion; further growth deferred.
- Implementation and local verification: Complete for this migration; see checks above.
- Operator deployment, runtime verification and timing results: TBD.

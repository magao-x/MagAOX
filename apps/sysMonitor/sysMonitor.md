\ingroup sysMonitor
# System diagnostics

`sysMonitor` reports Linux MD RAID health, can report NVIDIA GPU measurements,
and can check dedicated device IRQ placement alongside its existing CPU, disk,
RAM, and Chrony diagnostics. These collectors are read-only. They do not change
arrays, GPU clocks, IRQ affinity, cpusets, or services.

RAID discovery is enabled by default. GPU monitoring defaults to disabled. IRQ
checking is disabled when no IRQ groups are configured. Merge the appropriate
`config/sysMonRTC-diagnostics.conf` or `config/sysMonICC-diagnostics.conf` fragment
into the host's normal configuration to enable GPU/IRQ diagnostics and require
the host's expected RAID UUIDs. `config/sysMonAOC-diagnostics.conf` supplies the
AOC RAID expectations. The RTC/ICC fragments follow the dedicated groups in the merged
`rtSetup/*/move_irqs` scripts. NVIDIA IRQs deliberately remain outside these
groups because they now use system CPUs.

## GPU measurements

```ini
[gpu]
enabled=true
interval=5
timeout=2
# Optional: require these physical GPUs even before successful discovery.
# expectedUUIDs=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The collector queries `/usr/bin/nvidia-smi` once per polling interval, using the
normal application user. The interval and command timeout are positive seconds.
Boolean settings accept `true`, `false`, `1`, or `0`; other values are rejected.
Polling follows the application loop, so existing collection work can delay a
sample. No additional NVIDIA library is needed at build time.

Each discovered physical GPU has a numeric property named for its reported
NVIDIA index, such as `gpu_0` or `gpu_1`. These indices come from `nvidia-smi`,
not CSV row order or UUID sorting. The companion `gpu_0_info` or `gpu_1_info`
contains `uuid`, `pci`, and `status` so the current device association is visible.
UUID remains the stable internal identity, while telemetry identifies each GPU
by its reported index. The app logs index-to-UUID/PCI mappings at first successful
discovery and whenever a mapping changes. INDI property associations can change
when NVIDIA indices change.

| Numeric element | Units | Validity bit |
| --- | --- | --- |
| `temperature` | degrees C | 1 |
| `load` | kernel-active fraction, 0–1 | 2 |
| `memory_load` | memory read/write-active fraction, 0–1 | 4 |
| `memory_used` | MiB | 8 |
| `memory_total` | MiB | 16 |
| `clock_graphics` | MHz | 32 |
| `clock_sm` | MHz | 64 |
| `clock_memory` | MHz | 128 |
| `valid` | sum of valid field bits; 255 means all available | — |
| `sample_time` | Unix UTC seconds of the last successful device query | — |

Memory activity is distinct from allocated memory. Clocks are current reported
values; this collector does not request clock changes. `N/A` and unsupported
fields clear the corresponding validity bits while preserving other fields.

`gpu_status.status` reports disabled, unchecked, successful, partial, or failed
collection, including when no GPUs can be discovered. Failed reads clear validity
and retain the last device sample time; retained numbers must not be interpreted
as fresh measurements. INDI properties are read-only and non-healthy enabled
states are marked Alert. Failures and recovery are logged on status changes.

A missing known GPU retains invalid values at its last reported index until that
index belongs to another GPU. Reindexing invalidates an old property that no
longer has an owner; an index reused by another GPU publishes the new UUID and
measurements. The missing UUID remains in internal state and status logs even
if its former index is reused. An expected UUID that has never been discovered
has no known index, so it does not create a numbered INDI property or a GPU
telemetry sample. Devices whose former indices have been reclaimed are also
omitted from GPU sample vectors while their index is unknown. The overall
collection status and UUID-specific status logs still explain their absence.

The CSV interface is a dependency on the installed NVIDIA driver release. Check
the fields after driver upgrades. MIG instances, per-process GPU accounting,
power measurements, and automatic temperature alarms are outside this feature.

## Dedicated IRQ placement

```ini
[irq]
interval=5

[irq.camwfs]
match=edt
cpuset=camwfs_irq
# required=true
# requireEffective=true
```

Every group needs a unique name containing only letters, digits, or underscores;
`summary` is reserved. `match` is a literal substring of the interrupt action
names. Supply exactly one of `cpuset` or `cpus`. The latter accepts a CPU list such
as `45` or `45-47,109-111`; a named cpuset follows the framework's configured
cpuset mountpoint and its current `cpuset.cpus` contents.

At startup, the app logs each group's index in lexicographic name order for the
configured group set, its action-name match, cpuset or explicit CPU list, and
`required`/`requireEffective` policy. The first check logs the resolved expected
CPUs, or their unavailability. Later resolution after an unavailable cpuset, or
a change to the expected CPU set, is logged when it occurs.

The checker resolves all matching numeric IRQs on every poll. It compares sets,
so equivalent CPU-list spellings are accepted. Requested affinity must equal the
expected set. Effective affinity must be nonempty and contained within it; it
need not equal a multi-CPU requested set. Every matching vector is checked even
when its interrupt counters have not changed.

Groups are required by default. Set `required=false` only for a device whose
absence is expected during normal operation. A present optional device is still
checked and can warn. Missing or unreadable effective affinity warns by default.
`requireEffective=false` permits requested-affinity-only checking when the
effective-affinity file is absent; it does not suppress malformed data, a
disappearing IRQ, or an effective affinity outside the expected CPUs.

Read-only INDI properties provide:

- `irq_status`: overall `summary` and each group's verification result.
- `irq_warning.active`: On when a required group is unchecked, missing, or
  unverifiable under its policy, or any present group fails its check.

An optional missing group is shown as absent. Monitoring with no configured
groups is shown as disabled, and initial unchecked state is not reported as a
successful check. Detailed placement is written to sysMonitor fault logs: each
changed fault snapshot includes expected CPUs and the current IRQ numbers with
requested/effective CPU lists. Repeated identical faults are suppressed even
when their check times advance. Healthy transitions and recovery produce short
notices without affinity lists. There is no `irq_details` INDI property; use
`irq_status` for the current result and the app log for fault details.

Install `config/sysmon-warning.conf` in the configured `stateRuleEngine` rules
directory for RTC/ICC operator warnings and recovery notifications. It compares
the explicit warning switch; no changes to the rule engine are needed.

This checker covers configured dedicated devices. It does not audit every
ordinary IRQ against system CPUs, infer placement from historical counters, or
repair placement. Reads are a best-effort snapshot: device reloads can race a
check, and the next poll re-resolves the devices. Driver action-name or NIC-name
changes may require a configuration update; IRQ-number changes do not.

## Linux MD RAID health

```ini
[raid]
# These are the defaults; omit this section to use them.
enabled=true
interval=5

# Optional: require this array even when absent at startup.
# Replace the placeholder with the UUID from /sys/block/<name>/md/uuid.
# [raid.data]
# uuid=<kernel MD UUID>
```

The collector reads `/proc/mdstat` directly with the normal application user's
permissions. It runs every five seconds by default; `interval` must be finite
and positive. The first application poll checks immediately without waiting for
the interval. Polling follows the application loop, so other collection work can
delay a check. It discovers arrays on each complete snapshot without invoking
`mdadm` or changing array state. This covers Linux MD arrays visible in mdstat,
not storage-controller health or individual disks' SMART status.

For RAID1/4/5/6/10, configured/active member counts must agree with the `U`/`_`
availability map. Missing redundancy warns throughout a rebuild. Failed members
marked `(F)` also warn, including failed spares even when full redundancy remains.
Ordinary inactive arrays, broken arrays, unsupported states, and unreadable or
malformed snapshots warn. Spares, replacement members, write-mostly members, and
journal members do not warn merely because of their role.

Maintenance alone does not mean degradation: recovery, resync, check, repair,
reshape, and pending/delayed/remote work are reported without creating an alarm
when the array otherwise passes its health check. Read-only and auto-read-only
modifiers alone do not cause a degradation alarm. Active RAID0/linear arrays have
no redundancy counts. Recognized inactive `external:imsm` and `external:ddf`
metadata containers are normal bookkeeping; their inactivity does not warn, but
a failed component still does. Inactive external member arrays are not containers
and do warn.

Without any `[raid.NAME]` sections, monitoring uses mdstat alone and tracks each
observed MD name by a stable index for the current process. This mode cannot
recognize an array that is already absent at startup. An MD-name change cannot
establish physical identity: the old name remains missing and the new name
receives another index.

To require an array from startup, add a section such as `[raid.data]` with its
kernel MD `uuid`. The nonempty section suffix (`data` here) is the operator label
and may contain letters, digits, underscores, and hyphens. Each configured UUID
is required. Empty section headers with no keys are ignored by the configuration
framework; include the `uuid=` line, whose empty or invalid value is rejected.
The app accepts the standard hyphenated UUID
or its 32 hexadecimal digits, normalizes case, and rejects zero, malformed, or
duplicate UUIDs. Use the value from `/sys/block/<name>/md/uuid`. Colon-separated
`mdadm` UUIDs are not accepted: older 0.90 metadata can have different byte
ordering, so changing punctuation is not a reliable conversion.

The host fragments contain the UUIDs supplied on 2026-11-16, labeled from their
local mount points (`root` for `/`, with underscores replacing other slashes)
and confirmed swap roles:

| Host | Expected array labels |
| --- | --- |
| RTC / exao2 | `root`, `boot`, `boot_efi`, `data`, `swap` |
| ICC / exao3 | `root`, `boot`, `boot_efi`, `data`, `swap` |
| AOC / exao1 | `root`, `boot_efi`, `home`, `mnt_backup` |

The user confirmed that RTC md127 and ICC md126 are swap arrays. Both retain
their supplied UUIDs and are required under the `swap` label.
The AOC backup UUID is required even though its supplied mdstat snapshot reports
a broken array. Network mounts are covered on the host owning the underlying
array. Names in config comments record the supplied snapshot; UUIDs determine
identity. Update a configured UUID deliberately when replacing/recreating an array.

Any configured expectation enables UUID tracking for **all** discovered arrays.
Health still comes from mdstat, and identity comes from each array's
`/sys/block/<name>/md/uuid`. This requires readable, valid UUID attributes;
the attribute was introduced upstream for Linux 5.9. Native MD 1.0/1.2 metadata
is the primary supported case. Older kernels/modules or metadata that cannot
provide a usable UUID cannot satisfy this mode. See the
[Linux 5.9 MD source](https://github.com/torvalds/linux/blob/v5.9/drivers/md/md.c).

Configured arrays receive indices and unchecked status at startup. The first
check reports a configured UUID as missing if it is absent, including when
mdstat is empty or absent. Autodiscovery continues: unconfigured UUIDs are added
and monitored normally. A UUID retains its index through an MD-name change;
a replacement UUID at the same name receives a new index while the old UUID
remains missing. Expectation and discovery/mapping logs record the label, UUID,
current name, and level when available.

If identity cannot be established for any discovered array, the entire
collection becomes unknown and warns. Its partial inventory is ignored, retained
arrays become unavailable, and a later complete read can recover. This avoids
matching an expectation using an unverified name. There is no fallback to
name-based identity while expectations are configured.

Previously discovered arrays remain in the inventory as missing if they
disappear. This also applies to metadata containers: their disappearance warns
about inventory loss. A valid empty snapshot means no arrays only when there are
no configured or previously discovered arrays. A missing `/proc/mdstat` is benign
only in that same case; other read failures remain unknown and warn. Failed reads
or parsing never establish healthy state. The discovered inventory resets at
restart; configured expectations are recreated from configuration.

Read-only INDI properties provide:

- `raid_status.summary`: overall inventory/collection result.
- `raid_warning.active`: On when a configured or observed array is missing,
  faulty, degraded, inactive, or unverifiable, or collection fails. A recognized
  healthy metadata container and an initial no-array host without expectations
  do not warn.
- `raid_0`, `raid_1`, etc.: `valid`, `ok`, `degraded`, `configured`, `expected`,
  `active`, `flags`, and `check_time`. The first four are zero/one values;
  `configured` identifies an expected UUID, while `expected` remains the required
  **member count**. `flags` uses the bit values below. Counts are zero unless
  `Redundant` is set. Time is Unix UTC seconds of the actual check attempt, or
  zero before an attempt.
- `raid_0_info`, `raid_1_info`, etc.: the corresponding `name`, `level`, `label`,
  `uuid`, and human-readable `status`. Labels identify configured expectations;
  UUIDs are populated in UUID-tracking mode. An expected array that has not been
  discovered has no current MD name or level. UUID-tracking mode also clears
  these fields when an identity is missing or unavailable, avoiding a stale
  association if another array reuses its old name.

`ok` means the current assessment satisfies the monitoring policy; it is not
proof of redundant storage. An active RAID0/linear array or recognized healthy
metadata container can be OK. Use `valid` and `Redundant` when interpreting counts.
Changed faults log array identity, member descriptors, and the availability map
when present. Identical faults are suppressed, and recovery produces a short
notice. Changing maintenance percentages, ETA, and throughput do not generate
repeated diagnostic details or telemetry.

The `config/sysmon-warning.conf` examples add RAID warnings and recovery
notifications for `sysMonRTC`, `sysMonICC`, and `sysMonAOC` through the existing
stateRuleEngine warning switch rules. Install the rules in its configured rules
directory; RAID discovery itself does not require installing a config fragment.

## Telemetry and qualification

`telem_gpu` records the reported NVIDIA index, measurements, per-field validity,
collection/device status, and actual sample times. UUID and PCI strings are
omitted from new records; use the first-discovery and mapping-change logs to
identify the physical device. Samples without a known index are omitted.

`telem_irq` records only a vector of per-group `ok` flags and the actual
`check_time`. Vector positions follow the group indices logged at startup:

- `0`: unchecked, or the completed check failed the configured policy.
- `1`: a completed check is acceptable under the configured policy. This
  includes an absent optional device and requested-affinity-only checking when
  explicitly permitted.

Group names, status strings, IRQ numbers, and CPU lists are not repeated in new
IRQ telemetry. Startup and expectation-change logs describe the policy, and
changed-fault logs carry placement details. A first record, a flag-vector change,
or the existing telemeter maximum interval emits IRQ telemetry. Detail-only or
check-time-only changes do not force an extra record. Forced records carry the
latest actual check time; they do not claim a new check. GPU records retain
first/change/forced behavior and their actual sample times.

`telem_raid` records collection status, actual `check_time`, and each known
array's index, flags, expected count, and active count. Collection status describes
the read/parse outcome, independently of individual array health: `Unchecked=0`,
`Ready=1`, `ReadError=2`, `ParseError=3`, `Disabled=4`, and `NotPresent=5`.

| RAID flag | Bit value | Meaning |
| --- | --- | --- |
| `Valid` | 1 | The current array health assessment is usable |
| `OK` | 2 | The array satisfies the monitoring policy |
| `Degraded` | 4 | A redundant array has fewer active members than expected |
| `Missing` | 8 | A configured or discovered array is absent, or mdstat has disappeared |
| `Inactive` | 16 | An ordinary array is present but inactive |
| `Redundant` | 32 | Expected/active counts describe redundant members |
| `Container` | 64 | A recognized external metadata container |
| `Faulty` | 128 | A failed component or broken array is reported |

RAID telemetry omits names, labels, UUIDs, levels, member lists, and maintenance
progress. UUID expectations do not change its compact schema or add a configured
flag; startup expectation and discovery/mapping logs provide that context. It
records first, changed numeric state, and forced snapshots. Time-only or
detail-only changes do not force another record; forced records preserve the
latest actual check time. Retain RAID expectation, discovery/mapping, and
changed-fault logs with telemetry extracts. Array indices are local to the
current process and may change after restart.

Updated readers can decode older GPU/IRQ records because legacy schema field
positions remain available. Older binaries must be updated before reading the
new compact records or the new RAID event type. Retain startup, GPU/RAID mapping,
and IRQ/RAID expectation logs alongside telemetry, including when rotating files
or extracting a time window; a telemetry segment alone may not identify a
physical GPU, RAID array, or IRQ group.

Qualify the configuration on the instrument from the operator's workstation.
Compare measurements with normal readbacks, check warning behavior before and
after the normal IRQ setup procedure, and confirm recovery through normal reboot
and device availability changes. Compare loop timing with monitoring enabled and
disabled, including the writer's own timing such as `fg_timing.delta_aw`.
For RAID, compare the reported array inventory and health with the operator's
normal storage checks, including existing degraded/rebuilding arrays when
available, and verify that installed warning rules receive the warning switch.
Before enabling expected-array sections, have the operator obtain the real
kernel MD UUIDs from the monitored host and confirm that sysMonitor's application
user can read UUID attributes for every discovered array.

A GPU query deadline bounds userspace collection. A child stuck in uninterruptible
kernel sleep may remain pending after termination is requested; further GPU
queries are withheld until it is reaped. CPU/disk/IRQ/RAID monitoring continues.

\ingroup sysMonitor
# System diagnostics

`sysMonitor` can report NVIDIA GPU measurements and check dedicated device IRQ
placement alongside its existing CPU, disk, RAM, and Chrony diagnostics. Both new
collectors are read-only. They do not change GPU clocks, IRQ affinity, cpusets, or
services.

GPU monitoring defaults to disabled. IRQ checking is disabled when no IRQ groups
are configured. Merge the appropriate `config/sysMonRTC-diagnostics.conf` or
`config/sysMonICC-diagnostics.conf` fragment into the host's normal configuration
to enable them. These fragments follow the dedicated groups in the merged
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

Updated readers can decode older GPU/IRQ records because legacy schema field
positions remain available. Older binaries must be updated before reading the
new compact records. Retain startup, GPU mapping, and IRQ expectation logs
alongside telemetry, including when rotating files or extracting a time window;
a telemetry segment alone may not identify a physical GPU or an IRQ group.

Qualify the configuration on the instrument from the operator's workstation.
Compare measurements with normal readbacks, check warning behavior before and
after the normal IRQ setup procedure, and confirm recovery through normal reboot
and device availability changes. Compare loop timing with monitoring enabled and
disabled, including the writer's own timing such as `fg_timing.delta_aw`.

A GPU query deadline bounds userspace collection. A child stuck in uninterruptible
kernel sleep may remain pending after termination is requested; further GPU
queries are withheld until it is reaped. CPU/disk/IRQ monitoring continues.

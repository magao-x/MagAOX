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

Each physical GPU has a numeric property named `gpu_<uuid>`, with UUID hyphens
replaced by underscores and hexadecimal digits lowercased. Its companion
`gpu_<uuid>_info` contains `uuid`, `pci`, and `status`. UUIDs keep the association
stable when the driver's enumeration order changes.

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
collection, including when no GPUs can be discovered. A previously known or
explicitly expected GPU remains visible when missing. Failed reads clear validity
and retain the last device sample time; retained numbers must not be interpreted
as fresh measurements. INDI properties are read-only and non-healthy enabled
states are marked Alert. Failures and recovery are logged on status changes.

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
- `irq_details`: expected CPUs and current IRQ numbers with requested/effective
  CPU lists for each group.
- `irq_warning.active`: On when a required group is unchecked, missing, or
  unverifiable under its policy, or any present group fails its check.

An optional missing group is shown as absent. Monitoring with no configured
groups is shown as disabled, and initial unchecked state is not reported as a
successful check. Unchanged results do not repeatedly log warnings.

Install `config/sysmon-warning.conf` in the configured `stateRuleEngine` rules
directory for RTC/ICC operator warnings and recovery notifications. It compares
the explicit warning switch; no changes to the rule engine are needed.

This checker covers configured dedicated devices. It does not audit every
ordinary IRQ against system CPUs, infer placement from historical counters, or
repair placement. Reads are a best-effort snapshot: device reloads can race a
check, and the next poll re-resolves the devices. Driver action-name or NIC-name
changes may require a configuration update; IRQ-number changes do not.

## Telemetry and qualification

`telem_gpu` records identity, measurements, validity, and actual sample times.
`telem_irq` records logical groups, current IRQ numbers, expected/requested/effective
CPU lists, check status, and actual check times. Records are emitted first, when
state changes, and at the existing telemeter maximum interval. A forced record
of cached state retains the original measurement/check time.

Qualify the configuration on the instrument from the operator's workstation.
Compare measurements with normal readbacks, check warning behavior before and
after the normal IRQ setup procedure, and confirm recovery through normal reboot
and device availability changes. Compare loop timing with monitoring enabled and
disabled, including the writer's own timing such as `fg_timing.delta_aw`.

A GPU query deadline bounds userspace collection. A child stuck in uninterruptible
kernel sleep may remain pending after termination is requested; further GPU
queries are withheld until it is reaped. CPU/disk/IRQ monitoring continues.

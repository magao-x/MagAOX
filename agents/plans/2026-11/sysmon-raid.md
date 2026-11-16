# sysMonitor RAID monitoring

User request: "ok this was merged. To continue, I'd like to add monitoring of RAID status to sysMonitor. The main purpose is to provide operator warnings if a RAID is degraded. Telemetry will also be good. I think we can parse /proc/mdstat and auto-detect the arrays."

Status: RAID monitoring and expected-UUID follow-up implemented and validated on `jrmales/sysmon-raid`, based on merged `dev` at `00ca7378` (PR #456). Checkout `/home/jrmales/Source/MagAOX-working`; workstation `mx11`. No instrument commands or hardware changes are authorized or needed for offline implementation.

## Initial design

- Read `/proc/mdstat` directly on a configurable five-second cadence. Discovery is enabled by default (`raid.enabled=true`); `raid.interval` must be finite and positive. Use bounded reads and strict complete-snapshot parsing, with no mdadm commands or configuration writes.
- Auto-discover all named MD arrays. Preserve a stable per-process numeric index for each observed name, log its name/level mapping on discovery and changes, and retain disappeared arrays as missing until they return or sysMonitor restarts. Auto-discovery cannot recognize an array already absent at startup, or establish physical identity across an MD-name change.
- For RAID1/4/5/6/10, validate configured/active counts against the U/_ member map. Missing redundancy warns, including during rebuild. Failed members `(F)`, ordinary inactive arrays, broken arrays, unsupported/unverifiable states, and read/parse errors also warn. Spares and replacement/write-mostly/journal markers alone do not warn.
- Maintenance (`recovery`, `resync`, `check`, `repair`, `reshape`, pending/delayed/remote work) does not alone imply degradation. Read-only/auto-read-only are reported without a degradation alarm. Active RAID0/linear arrays have no redundancy counts; recognized inactive `external:imsm`/`external:ddf` metadata containers are bookkeeping, not failed data arrays. Other inactive external member arrays still warn.
- A valid empty snapshot means no arrays. Missing `/proc/mdstat` before any array has been observed means MD is not present and does not raise a degradation warning; other read failures remain explicit. Once an array has been observed, disappearance of the array or the file warns.
- Publish overall `raid_status`, per-array read-only status/health/counts, and `raid_warning.active` for stateRuleEngine. Add warning examples for the system monitors. Log changed fault details and brief recovery; avoid repeated details and progress/ETA noise.
- Add numeric-only `telem_raid` with collector status, actual check time, and per-array index/flags/configured/active counts. Flags distinguish valid, OK, degraded, missing, inactive, redundant, metadata container, and failed component states. Counts apply only to redundant arrays. Preserve startup/discovery mappings with telemetry extracts. Record first, changed numeric state, and forced snapshots, retaining the actual check time.
- Keep existing GPU/IRQ telemetry formats and behavior unchanged.

## Expected UUID follow-up — 2026-11-16

User request: "ok I see the limitation now. Let's have a config section where we identify the expect RAIDs, maybe by UUID so we don't get confused by simple config changes. We should also continue auto discovery and track new ones or not-configured ones. But this way an expected but missing array gets reported ASAP."

- Add named `[raid.NAME]` sections with a required `uuid`, using the kernel MD UUID from `/sys/block/<name>/md/uuid`. Validate labels and reject malformed, zero, or duplicate UUIDs. Accept case-insensitive dashed or compact hexadecimal text and normalize to lowercase dashed form. Reject mdadm colon syntax because metadata 0.90 can use different word byte order. The existing configuration framework drops completely empty section headers; document that an expectation requires the `uuid=` key, and reject empty values or populated expectation sections missing that key.
- Seed configured identities and their numeric indices at startup, logging label/UUID/index even if an array never appears. Remain unchecked until the first immediate application poll; a complete empty inventory or missing mdstat then marks each expected identity Missing and raises the existing operator warning. Disabling RAID collection also disables expectation warnings.
- Configured expectations enable strict UUID tracking for all discovered arrays, including unconfigured arrays. Match by UUID, retain indices through MD-name changes, and log changed mappings. A replacement UUID at the same MD name is a new identity and does not satisfy the original expectation. Retain absent expected and discovered identities.
- With no expectations, preserve the existing mdstat-only name-based discovery behavior. Strict mode requires a readable unique nonzero kernel UUID for every discovered array. If any lookup fails or any UUID is invalid/duplicated, reject the partial inventory and mark retained identities unavailable. Never infer a match from an unverified name. A later complete read recovers normally.
- Strict UUID mode targets persistent native Linux MD arrays, including the supplied metadata 1.0/1.2 examples. Upstream provides the read-only UUID attribute from Linux 5.9 (vendor backports can differ). External metadata containers may expose zero kernel UUIDs, and metadata=none has no persistent identity; those are not suitable for configured UUID expectations.
- Add `configured` to per-array numeric INDI properties and `label`/`uuid` to companion information. Keep `expected` as the redundancy member count. No telemetry schema change: startup expectation/discovery mappings associate numeric indices with UUIDs and labels; first/change/forced recording remains unchanged.
- Save the user's exao1/exao2/exao3 mdstat samples as offline fixtures. They contain 4/5/5 arrays; exao1's md124 is broken with five failed components and `[5/0] [_____]`, and must warn. The user subsequently supplied all 14 kernel UUIDs and mount points; host configuration fragments now require those identities as described below.

Identity sources: [kernel MD UUID interface](https://docs.kernel.org/admin-guide/md.html#md-devices-in-sysfs), [kernel UUID byte formatting](https://docs.kernel.org/core-api/printk-formats.html#uuid-guid-addresses), upstream [v5.8 MD source](https://github.com/torvalds/linux/blob/v5.8/drivers/md/md.c) versus [v5.9 MD source](https://github.com/torvalds/linux/blob/v5.9/drivers/md/md.c), and mdadm [native 1.x](https://github.com/md-raid-utilities/mdadm/blob/main/super1.c) / [native 0.90](https://github.com/md-raid-utilities/mdadm/blob/main/super0.c) UUID formatting.

Follow-up validation: library and application builds passed, as did all eight focused suites (62 cases): RAID parser/identity (11), application diagnostics/configuration/lifecycle (21), RAID telemetry (5), existing sysMonitor parsing (6), GPU parser/telemetry (5 each), IRQ parser (5), and IRQ telemetry (4). Final application rebuild and 21-case diagnostics rerun passed after review corrected stale name/level display for missing UUIDs. The UUID follow-up changes no telemetry schema or generator. Formatting and `git diff --check` pass. Local output is under `/tmp/sysmon-raid-uuid-validation` and `/tmp/sysmon-raid-uuid-final`. All checks used saved mdstat text, synthetic sysfs files, or injected snapshots; no real MD devices were interrogated.

## Host UUID configuration — 2026-11-16

The user supplied read-only `/sys/block/md*/md/uuid` and `df -h` results for RTC/exao2, ICC/exao3, and AOC/exao1. Populate the RTC/ICC fragments and add `config/sysMonAOC-diagnostics.conf` with these exact kernel UUIDs. Use mount-based labels (`/` becomes `root`; other path separators become underscores) and `swap` for the user-confirmed swap arrays. Names in comments refer only to the supplied snapshot, not identity matching.

| Host | Label → observed array / mount |
| --- | --- |
| RTC | `root` → md126 `/`; `boot` → md125 `/boot`; `boot_efi` → md123 `/boot/efi`; `data` → md124 `/data`; `swap` → md127 |
| ICC | `root` → md127 `/`; `boot` → md125 `/boot`; `boot_efi` → md124 `/boot/efi`; `data` → md123 `/data`; `swap` → md126 |
| AOC | `root` → md127 `/`; `boot_efi` → md126 `/boot/efi`; `home` → md125 `/home`; `mnt_backup` → md124 `/mnt/backup` |

All 14 arrays are required from startup. The user confirmed that RTC md127 and ICC md126 are swap arrays; their labels are now `swap`, with the supplied UUIDs unchanged. The backup array remains expected despite its earlier broken state. Remote network mounts do not create duplicate local RAID expectations. AOC's fragment adds RAID settings only, preserving existing GPU/IRQ configuration when merged.

Validation: independent review and a structured config check matched all 14 UUIDs and labels to the user's supplied data, checked uniqueness and normalized syntax, and confirmed the enabled five-second cadence. The existing application diagnostics suite passed all 21 cases / 1048 assertions with the populated RTC/ICC fragments (`/tmp/sysmon-raid-host-config-tests.log`). `git diff --check` passed. No binaries, installed configs, services, or instrument state were changed for this configuration-only follow-up.

## Sources

- [Linux MD documentation](https://docs.kernel.org/admin-guide/md.html): array state, redundancy, metadata and maintenance semantics.
- [Kernel MD status implementation](https://github.com/torvalds/linux/blob/master/drivers/md/md.c): mdstat framing, array/member states and maintenance output.
- [RAID5 status implementation](https://github.com/torvalds/linux/blob/master/drivers/md/raid5.c): configured/active counts and U/_ availability map.
- [mdadm reference parser](https://github.com/md-raid-utilities/mdadm/blob/main/mdstat.c) and [mdmon documentation](https://github.com/md-raid-utilities/mdadm/blob/main/mdmon.8): external metadata containers versus member arrays.

## Documentation and operator configuration

The application guide describes default-on discovery, the distinction between
degradation and maintenance, read/parse failures, and the startup inventory
limitation. Disappearance of any previously discovered name, including a metadata
container, is an inventory-loss warning. A recognized inactive container is
otherwise acceptable unless it reports a failed component.

The documented INDI interface is `raid_status.summary`, `raid_warning.active`,
and stable per-process `raid_N` / `raid_N_info` properties. Numeric properties
contain `valid`, `ok`, `degraded`, `expected`, `active`, `flags`, and `check_time`;
the companion contains `name`, `level`, and `status`. Compact telemetry flags are
Valid=1, OK=2, Degraded=4, Missing=8, Inactive=16, Redundant=32, Container=64, and
Faulty=128. Counts are zero without Redundant. OK means policy acceptance and can
include active non-redundant arrays or healthy metadata containers. Mapping logs
must accompany extracted telemetry; indices are not identities across restarts.

RTC/ICC/AOC configuration fragments explicitly show the default
`[raid] enabled=true, interval=5` settings and their expected host UUIDs. The warning-rule example adds
`swVal` rules for `sysMonRTC`, `sysMonICC`, and `sysMonAOC`, targeting
`raid_warning.active=On`; no stateRuleEngine code change is needed. These are
repository examples only: no installed configuration or running service was
changed. Validation results are recorded below.

## Validation

Fixtures cover healthy/degraded maintenance, failed spares, inactive/broken/read-only states, external containers, non-redundant arrays, malformed/truncated input, discovery/disappearance/recovery, invalid configuration, and first/change/forced recording. A lifecycle regression also checks that changing maintenance percentages is silent while a changed faulty member produces a log without another numeric telemetry record.

Final library and sysMonitor builds passed. All eight focused suites passed: RAID parser (6 cases), RAID telemetry (5), sysMonitor diagnostics (16), existing sysMonitor parsing (6), GPU parser (5), GPU telemetry (5), IRQ parser (5), and IRQ telemetry (4). All eight Python generator tests and all 87 generated logger cases passed. `git diff --check` is clean.

Two test fixtures were corrected during validation: use the INDI string getter that preserves whitespace, and force FlatBuffers defaults in malformed floating-point fixtures so `-ffast-math` cannot erase NaN before validation. Both corrected suites passed on rerun. Production time validation uses `mx::math::isFinite`.

No instrument app startup or actual RAID management commands were run. Application tests use injected snapshots, private logs, and a serialization-only INDI driver. Builds use the normal optimization flags; app/test links add `LDFLAGS=--coverage` for the workstation's existing coverage-instrumented archives. Local validation output is in `/tmp/sysmon-raid-validation` and `/tmp/sysmon-raid-recheck`.

## Changed files

- Application: `apps/sysMonitor/sysMonitor.hpp`, `raidMonitor.hpp`, and `Makefile`.
- Application tests: `apps/sysMonitor/tests/raidMonitor_test.cpp`, `sysMonitorDiagnostics_test.cpp`, and captured `tests/fixtures/exao1.mdstat`, `exao2.mdstat`, and `exao3.mdstat` under the application directory.
- Operator guide/configuration: `apps/sysMonitor/sysMonitor.md`, `config/sysMonRTC-diagnostics.conf`, `config/sysMonICC-diagnostics.conf`, `config/sysMonAOC-diagnostics.conf`, and `config/sysmon-warning.conf`.
- Telemetry: `libMagAOX/logger/types/telem_raid.hpp`, `types/schemas/telem_raid.fbs`, `types/telem.cpp`, `logCodes.dat`, and `logMeta.cpp`.
- Logger tests: `libMagAOX/logger/tests/raidTelemetry_test.cpp`, `generateTemplatedCatch2Tests.py`, and `test_generateTemplatedCatch2Tests.py`.
- Build/test registration: `libMagAOX/Makefile`, `tests/Makefile.one`, `tests/tests.list`, and `tests/groups.dox`.
- Engineering record: `agents/plans/2026-11/sysmon-raid.md`.

## Operator follow-up

Install updated sysMonitor/logger binaries, merge each host's populated diagnostic fragment into its normal configuration, and merge the example RAID rules into stateRuleEngine's configured rules directory. Confirm the application user can read UUID attributes for every discovered array. Compare a real host's inventory and health with the read-only INDI output. Keep expectation and discovery logs with extracted telemetry. Without configured expectations, auto-discovery still cannot identify an array already absent when sysMonitor starts.

# sysMonitor RAID monitoring

User request: "ok this was merged. To continue, I'd like to add monitoring of RAID status to sysMonitor. The main purpose is to provide operator warnings if a RAID is degraded. Telemetry will also be good. I think we can parse /proc/mdstat and auto-detect the arrays."

Status: implemented and validated on `jrmales/sysmon-raid`, based on merged `dev` at `00ca7378` (PR #456). Checkout `/home/jrmales/Source/MagAOX-working`; workstation `mx11`. No instrument commands or hardware changes are authorized or needed for offline implementation.

## Design

- Read `/proc/mdstat` directly on a configurable five-second cadence. Discovery is enabled by default (`raid.enabled=true`); `raid.interval` must be finite and positive. Use bounded reads and strict complete-snapshot parsing, with no mdadm commands or configuration writes.
- Auto-discover all named MD arrays. Preserve a stable per-process numeric index for each observed name, log its name/level mapping on discovery and changes, and retain disappeared arrays as missing until they return or sysMonitor restarts. Auto-discovery cannot recognize an array already absent at startup, or establish physical identity across an MD-name change.
- For RAID1/4/5/6/10, validate configured/active counts against the U/_ member map. Missing redundancy warns, including during rebuild. Failed members `(F)`, ordinary inactive arrays, broken arrays, unsupported/unverifiable states, and read/parse errors also warn. Spares and replacement/write-mostly/journal markers alone do not warn.
- Maintenance (`recovery`, `resync`, `check`, `repair`, `reshape`, pending/delayed/remote work) does not alone imply degradation. Read-only/auto-read-only are reported without a degradation alarm. Active RAID0/linear arrays have no redundancy counts; recognized inactive `external:imsm`/`external:ddf` metadata containers are bookkeeping, not failed data arrays. Other inactive external member arrays still warn.
- A valid empty snapshot means no arrays. Missing `/proc/mdstat` before any array has been observed means MD is not present and does not raise a degradation warning; other read failures remain explicit. Once an array has been observed, disappearance of the array or the file warns.
- Publish overall `raid_status`, per-array read-only status/health/counts, and `raid_warning.active` for stateRuleEngine. Add warning examples for the system monitors. Log changed fault details and brief recovery; avoid repeated details and progress/ETA noise.
- Add numeric-only `telem_raid` with collector status, actual check time, and per-array index/flags/configured/active counts. Flags distinguish valid, OK, degraded, missing, inactive, redundant, metadata container, and failed component states. Counts apply only to redundant arrays. Preserve startup/discovery mappings with telemetry extracts. Record first, changed numeric state, and forced snapshots, retaining the actual check time.
- Keep existing GPU/IRQ telemetry formats and behavior unchanged.

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

RTC/ICC configuration fragments explicitly show the optional default
`[raid] enabled=true, interval=5` settings. The warning-rule example adds
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
- Application tests: `apps/sysMonitor/tests/raidMonitor_test.cpp` and `sysMonitorDiagnostics_test.cpp`.
- Operator guide/configuration: `apps/sysMonitor/sysMonitor.md`, `config/sysMonRTC-diagnostics.conf`, `config/sysMonICC-diagnostics.conf`, and `config/sysmon-warning.conf`.
- Telemetry: `libMagAOX/logger/types/telem_raid.hpp`, `types/schemas/telem_raid.fbs`, `types/telem.cpp`, `logCodes.dat`, and `logMeta.cpp`.
- Logger tests: `libMagAOX/logger/tests/raidTelemetry_test.cpp`, `generateTemplatedCatch2Tests.py`, and `test_generateTemplatedCatch2Tests.py`.
- Build/test registration: `libMagAOX/Makefile`, `tests/Makefile.one`, `tests/tests.list`, and `tests/groups.dox`.
- Engineering record: `agents/plans/2026-11/sysmon-raid.md`.

## Operator follow-up

Install updated sysMonitor/logger binaries and merge the example RAID rules into stateRuleEngine's configured rules directory. Compare a real host's inventory and health with the read-only INDI output. Keep discovery logs with extracted telemetry. Auto-discovery still cannot identify an array already absent when sysMonitor starts; an expected-array policy can be added separately if required.

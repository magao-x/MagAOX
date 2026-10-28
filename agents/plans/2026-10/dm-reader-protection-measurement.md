# DM reader-protection shadow measurement

## Authorized objective

On 2026-10-27 the user requested a test using the new controller completion
signal to quantify the required reader protection while testing the publisher
on the instrument. This follows the
[controller completion implementation](dm-controller-completion.md). All agent
work remains on the workstation; the operator performs instrument actions.

The measurement must preserve current operational and simulated decisions.
Actual main-command software completion is observed; virtual extra commands
do not have physical completion events. Replacing any protection budget or
selecting a physical/early-main reserve is outside this change.

## Design and endpoints

Three independent fixed-record SPSC handoffs capture:

1. The primary identity and exact monotonic observation used by
   `readSimulationPrimary()`.
2. The combined identity, named primary trace linkage and exact monotonic
   observation used by `readSimulationCombined()`, together with the current
   modeled budget `delivery + keepout + guard`.
3. A coherent dedicated controller completion record and its monotonic read
   time, captured by an optional ordinary-priority `completionMonitor`.

The first two reuse existing metadata snapshots while the simulation already
owns its cycle mutex. The completion callback acquires no simulation lock.
Callbacks perform fixed captures only; joining, allocation, statistics, INDI
and telemetry serialization run in ordinary app logic.

The consumer joins the exact combined inode/counter/raw writetime against the
controller record, then the named primary inode/counter against retained
coherent primary evidence. Primary publication must precede the combiner trace
start in the producer clock; producer timestamps are never subtracted from
local monotonic timestamps. Captured historical identities, rather than the
live stream's latest counter, are matched. Bounded pending storage supports
either capture arrival order, with expiry after the captured queues have all
received a bounded drain.

For primary observation P, combined observation C, controller entry S, controller
completion F and completion observation R, define B = max(P, C). Retain:

| Population | Definition |
| --- | --- |
| completionOffset | F − B, signed |
| requiredHold | max(0, F − B) |
| observedHold | max(0, R − B) |
| processing | F − S |
| notification | R − F |
| modelSlack | frozen budget − (F − B), signed |

The signed offset is essential: a negative result exposes a combined/primary
observation delayed until after controller completion. Clamping all evidence
to zero would hide that behavior. `requiredHold` describes actual remaining
software reads; `observedHold` includes delivery of the new notification and
is relevant to a future gate that waits to inspect it. Neither includes a new
qualified safety reserve.

All populations use seconds and independent count/variance/coverage semantics
from the existing shared timing accumulator. Latest INDI scalar maxima use
explicit `Us` suffixes. Lifetime maxima and exact provenance for the sample
with the greatest required hold survive rolling-window expiry and join resets.
`overBudget` counts observed signed offsets greater than their own frozen
budget. The default comparison budget is 155 us, not an optimized setting.

## Lifecycle, loss and configuration

Each source has its own queue. A global join generation conservatively revokes
old queued/pending identities after attachment or publisher/session/epoch
changes. Quality separately records source resets, stale generations, busy,
unavailable, error and unqualified records; incoherent/invalid captures,
identity mismatches, duplicate/gapped observations, pending eviction/expiry,
missing peers and per-source queue drops are visible. Completion age is
qualified at capture (R − F), independently of ordinary consumer queue delay.

The feature is off by default. A nonempty `completionMonitor.shmimName` enables
it only with `simulation.enabled=true` and enabled simulation diagnostics.
The optional monitor uses standard helper macros, ordinary priority zero and
no initial-existing-image callback. The dedicated completion input must be a
16-word uint64 CPU stream. Configured names remain distinct from DM inputs,
destinations, auxiliary sources and synco's schedule output.

Read-only `reader_timing`/`reader_timing_quality` expose rolling populations and
quality. A trailing optional `reader_timing` table extends `telem_dmsynco`,
retaining old-record compatibility and adding exact integer worst-case
identities/timestamps. The publisher itself still has no new telemetry schema;
this consumer test requires rebuilding synco and `logdump`.

## Implementation and verification record

Implementation includes common result types, the shared diagnostic collector,
the optional synco observer, read-only INDI, compatible telemetry, focused
collector/app/logger tests and standard build/test registration. The example
[`dmtweeter_synco_reader.conf`](../../../apps/dmSyncopation/dmtweeter_synco_reader.conf)
and [operator procedure](../../../apps/dmSyncopation/doc/reader-protection-test.md)
describe setup and paired capture.

The collector suite is registered in `tests/tests.list` and grouped under
`app_dev_unit_tests` in `tests/groups.dox`.

Local verification:

- Collector: **287 assertions / 15 cases**, including all arrival orders across
  updates, signed negative offsets, capture freshness versus delayed consumer
  update, errors/replay/reset/gaps, loss/retention, one primary with multiple
  combined outputs, and 512 exact joins from three concurrent producers.
- Full app integration: **8,301 assertions / 62 cases**, including **940
  assertions / 6 new cases**. Real callbacks match coherent records and publish
  read-only INDI. Serialized shadow state, command pixels/counters and semaphore
  populations remain unchanged by completion processing. A completion callback
  returns while the simulation mutex is held. Configuration, bad layouts,
  synthetic/busy/stale/error input and lifecycle changes are covered.
- Telemetry: **3,669 assertions / 23 cases**, including legacy compatibility,
  binary/JSON/text round trips with exact 64-bit evidence, malformed metrics and
  provenance, expired windows and lifetime extrema across source epochs.
- Existing pure shadow-model regression: **53,563 assertions / 37 cases**.
- Combined tested populations total **65,820 assertions / 137 cases**. Full
  touched C++ files pass clang-format and whitespace checks. New guide/plan
  local links resolve.

Normal library/PCH, `dmSyncopation` and `utils/logdump` builds all pass. The
library dispatch objects now explicitly depend on the changed `dmsynco` and
reader-result headers: updating a verifier/formatter after schema generation
must rebuild `logMeta.o`/`logMap.o`, not just the PCH. Fresh dispatch and binary
schema objects are used by the telemetry tests. No instrument commands,
installation, runtime configuration changes or hardware qualification were
performed by the agents.

## Interpretation and follow-up

Preserve operating settings and compare simulation with the completion
publisher enabled both with and without this observer, then make a longer
capture to retain rare extrema. Existing camera, primary, combined and DM
populations help identify added observer load. Missing notices can bias maxima
downward, so report the change in all quality counters alongside timing.

This is observed software read completion, with best-effort identity association
under existing ImageStreamIO/dmcomb publication conventions. It does not prove
atomic token/pixel provenance or physical mirror settling. A measured maximum
does not establish a hard bound under incomplete coverage. Any subsequent
replacement of a fixed hold must retain dmcomb-consumption, copy, early-main
and configuration/FSM protection, with its own instrument qualification.

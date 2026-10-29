# Signal-qualified reader readiness and three-step shadow test

## Authorized objective

On 2026-10-29 the operator selected `dmtweeter_read_done` as the reader-readiness
signal and requested evaluation of 6 kHz total cadence at a 2 kHz main rate.
The selected milestone is **shadow at three steps first**, not a live scheduler
revision or permission for agents to operate hardware. This follows the
[worker and observer measurements](bmc-sdk-worker-reader-release.md#16-fifo30--cpu57-capture-20261029t145715z).

The measured release requirement is about 20 us, with a current-run maximum
25.708 us and earlier observed maximum 28.303 us. Release observation on the
verified FIFO30 / CPU57 monitor takes about 3.681 us after the endpoint; its
current maximum is 8.086 us, with an older lifetime maximum 22.172 us. SDK
service remains about 104 us and the full callback about 131 us. Earlier
channel clearance does not make the SDK accept another command sooner.

## Fixed-grid geometry and independent constraints

At a 500 us main period, three total steps have active trigger targets at
166.667 and 333.333 us. With the existing 10 us copy allowance their copy
starts are 156.667 and 323.333 us. Existing early-main 20 us, observation 5 us,
visibility 1 us, passive lead 20 us and preparation reserve 2 us are unchanged.
The next passive copy targets 472 us and starts 462 us.

The last active command therefore has about 128.667 us before the next passive
copy starts, but only **141.667 us before the earliest modeled next main**.
Those are different constraints. An early read release can resolve the first
without resolving whole-backend occupancy before the next main.

The legacy full timing bundle requires a 193 us slot: 155 us reader hold,
10 us copy, 26 us tail and 2 us reserve. It remains unchanged for ordinary
shadow and live scheduling. The new shadow mode separates:

- Actual-main reader clearance: an exact qualified release observed by R,
  plus the existing configured guard, must precede the fixed copy start.
- Virtual auxiliary reader duration: explicitly assumed 30 us after its
  hypothetical trigger, plus the existing 20 us guard.
- Whole backend occupancy: explicitly assumed **155 us**, including delivery
  and callback retirement, independently of reader clearance. Previous backend
  work must fit before an active target, and a new active command must retire
  before the earliest modeled next main.

Keeping the conservative backend assumption at 155 us makes the nominal final
three-step slot fail its backend wraparound test: 333.333 + 155 exceeds 475 us.
The reader/copy geometry can fit while full-cycle backend geometry does not.
Do not silently lower backend occupancy or the undecided early-main reserve to
turn that failure into a successful 6 kHz claim. Explicit smaller backend
assumptions are supported for counterfactual shadow tests, not qualified as
hardware bounds. A same-frame backend retirement measurement or a separately
reviewed bound is needed before interpreting all three slots as deliverable.

## Actual release evidence

The release monitor copies a fixed coherent record with its local observation
time R and attachment generation into a preallocated SPSC handoff. It does not
take the simulation cycle mutex and does not wait for ordinary app logic. All
records, including incoherent/busy/unqualified evidence, are transported so
revocation is not hidden by filtering for successes alone.

The cycle owner drains a bounded queue population before primary, combined or
ordinary advance evaluations, holding one future-R record for a later update.
Each call handles at most 64 records, including a carried lookahead, so a
diagnostic-capacity backlog cannot become an unbounded monitor callback.
Attachment changes, model resets and transport loss fence queued evidence by
generation; a second reset/loss check closes readiness if either races the
drain. Queue allocation occurs only before producer startup, never during a
camera-rate/model reset.

The pure model retains bounded exact combined-input identities and release
history. Qualified readiness requires readers protocol, released state, a
genuine coherent current publisher/session/epoch, exact combined inode/counter/
raw writetime, consistent monotonic endpoints and a release observation available
before the fixed decision. R is used for availability, not the earlier endpoint
F or wait-return W. A late release cannot retrospectively turn an already
skipped occurrence into a successful copy, and skips never move later targets.

Invalid source evidence, unexpected progression or a source reset revokes
clearance. SDK failure is independent: finished reads are still a real endpoint,
but a backend fault closes further hypothetical active operations until recovery.
A later SDK-result enrichment of the same release is not another readiness
event and does not clear a newer or virtual job. A forward publisher restart
can establish a new epoch; backward publisher replay cannot.

Only real main commands produce physical release records during this test.
Virtual auxiliaries have no observed completion, so the explicit virtual-reader
and backend assumptions remain after each hypothetical auxiliary trigger. A
main release must never be reused as acknowledgement of a nonexistent virtual
auxiliary command.

Timestamp-ordered release observations advance already due slots strictly
before their own R, then apply their new readiness or revocation. This preserves
an earlier valid decision when an invalidation arrives later, while an
invalidation before a copy deadline closes that slot. A late event never
reclassifies an already counted occurrence.

Actual main arrivals are also checked against modeled backend ownership after
virtual readers clear. Backend conflicts have a distinct overlap window and
exact backend endpoint. The last active backend retains its own provenance,
because passive step-zero preparation can finish while that earlier operation
is still busy. Reset/silence clears both virtual ownership states. A reader
release or later passive copy cannot hide an auxiliary-backend/main collision.

## Configuration, readbacks and operator sequence

The new startup-only controls are:

| Key | Default | Meaning |
| --- | ---: | --- |
| `simulation.readerReleaseReady` | false | Use exact observed actual-main release in the shadow model. Requires simulation and a configured release monitor. |
| `simulation.virtualReaderUs` | 30 | Assumed virtual auxiliary reader duration before the unchanged guard. |
| `simulation.backendUs` | 155 | Assumed complete backend occupancy; independent of the read release. |

Legacy mode and live behavior remain unchanged. Readiness mode adds separate
reader-cycle and backend-cycle feasibility, raw release evidence and quality,
and per-step `releaseMissing`, `releaseLate`, `backendBusy` and `backendWrap`
skip reasons. Existing `readerHold` covers remaining modeled virtual reader
ownership. Overall feasibility reports the conjunction of the independent
constraints, while per-step outcomes remain evaluable when the whole requested
cycle cannot fit.

The optional telemetry readiness table distinguishes real-main evidence from
virtual assumptions, preserves exact integer identity/time fields and retains
old-record compatibility. The ordinary joined `reader_release_timing` summaries
remain diagnostics and are never the readiness authority.

The operator rebuilds dmSyncopation and logdump, keeps operational synco Off,
and retains the working controller, camera, loop, RTC latency and CPU placement.
The SDK worker remains on spare56 and the RT release monitor on spare57.
The additive three-step profile requests the candidate rate without changing
delivery, keepout, guard, early-main or copy settings. Begin with the new
readiness mode at two steps to compare against the existing model, then request
three and retain per-reason outcomes and both feasibility values. Continue to
use the bounded simultaneous capture with before/after controller diagnostics.

No agent commands are executed on the instrument. Actual auxiliary triggering,
the revised fixed-grid live scheduler and mode-3 primary-semaphore wake routing
remain separate implementation and qualification milestones. A raw 6 kHz
`dmtweeter_read_done` counter rate still represents 2 kHz commands under the
three-update protocol; it is not evidence of 6 kHz physical DM submissions.

## Implementation and verification

Implemented the raw fixed-record transport, per-call limit of 64 including a
carried lookahead, attachment/drop generation fencing, event-time qualification,
independent virtual ownership and backend conflict detection. The live scheduler
and default legacy shadow mode are unchanged. New configuration, readbacks,
backward-compatible telemetry and the additive candidate profile are described
in the [operator procedure](../../../apps/dmSyncopation/doc/readiness-shadow-test.md).

Deterministic tests exercise exact availability plus guard, releases arriving
before combined association or after deadlines, revocation before/after an
already committed decision, source replacement/replay, queue overflow/reset,
nonblocking observer capture, bounded draining including future lookahead,
genuine vs virtual ownership, SDK faults/enrichment and fixed-grid independent
skips. Backend conflicts are tested both after reader clearance and after a
subsequent passive copy has replaced latest-copy provenance; interval ends
remain exclusive. Three-step tests preserve the 155 us final-slot rejection
and separately exercise explicit smaller assumptions and nanosecond boundaries.

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| Pure shadow engine | 53,767 | 49 |
| Synco application integration | 11,347 | 77 |
| Synco telemetry and compatibility | 6,421 | 33 |
| **Total** | **71,535** | **159** |

Fresh library/PCH and generated schema/dispatch, normal dmSyncopation and
logdump builds pass. Telemetry tests cover optional-field legacy compatibility,
exact full-width JSON provenance, exclusive skip accounting and malformed
readiness/backend evidence. Whole changed C/C++ files pass clang-format;
whitespace and operator-guide Bash syntax/local links are checked. These are
workstation checks only; no instrument action or 6 kHz hardware qualification
was performed by the agents.

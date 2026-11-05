# Production providers

Production mode uses the counter-publication scheduler qualified by the finite
zero trials, with changing, independent contributions from applications on the
same RTC. It supports one through four steps, including passive-only `N=1`;
four steps at 2 kHz main request 8 kHz total. `dmSpeckle` is the first application using the shared
`dev::dmSyncopationProducer<derivedT>` helper. The implementation and verification
record is in the [production plan](../../../agents/plans/2026-11/dm-syncopation-production-providers.md).

## Command ownership

Each producer owns one CPU-backed, two-dimensional float32 image named
`<destination>_aux`, with the same geometry and command units as its dmcomb
destination. For example, `dm01disp07_aux` feeds `dm01disp07`. Configure different
destinations for different producers. A producer writes only its auxiliary
source; synco copies into the numbered dmcomb input and controls auxiliary
counter publications.

The main publisher continues to control step 0. At 2 kHz main and four steps,
the interleaved source counter targets are:

```text
|----------------+----------------+----------------+----------------|
0               125              250              375              500 us
main cnt0++     aux cnt0++       aux cnt0++       aux cnt0++       main cnt0++

               producer complete     snapshot/copy/checks       target
                      |------------------------|-----------------|
                 input cutoff                          auxiliary cnt0++
```

Step 0 performs the same ready-copy operation, preserving destination counters;
the genuine main update triggers dmcomb. Copies require prior-reader clearance
and must fit before the configured passive-copy cutoff. A previous read-done
signal clears previous readers; it does not prevent a new, early main from
starting. A main arriving before preparation begins leaves the prior
contributions in place. Counter checks can detect an arrival during copying,
but cannot undo pixels already read.

## Independent input deadlines

Each advertised slot has an immutable identity and input cutoff. For each
producer, synco selects a complete matching command published by that cutoff,
or retains the last contribution copied to that producer's destination. A miss
from one producer does not suppress another producer's eligible update. A late
command tagged for a past slot is discarded rather than applied to a later slot.

Every selected contribution is copied, including retained and unchanged values.
There is no pixel comparison or no-copy HOLD branch. Producers explicitly
submit zero to remove a contribution. Stopping a producer retains its last
contribution. Shared reader, primary/combined identity, clock, cancellation,
copy-window and publication guards can still suppress a whole operation.

Copy completion and trigger/consumption are separate outcomes. After a passive
copy completes, suppressing a later auxiliary trigger alone does not restore
the preceding pixels. Retention follows the completed destination copy; it
does not require a per-producer physical-DM acknowledgement.

The qualified 6 us allowance belongs to synco's final publication admission.
It does not extend a producer cutoff, shift subsequent targets, or guarantee
an exact counter-store time if the process is preempted inside the update API.

## Shared-memory protocol

The helper handles these records; applications should use its request and
publish interfaces instead of creating their own transport implementation.

| Stream | Owner | Meaning |
| --- | --- | --- |
| `<instance>_timing` | synco | Version-2 schedule, validity/expiry, epoch/generation and future slot cutoffs/targets. |
| `<destination>_aux` | producer | One complete float32 contribution. |
| `<destination>_aux_ready` | producer | Versioned command identity and exact source inode/counter, producer session/sequence and ready timestamp. |
| `<destination>_aux_ack` | synco | Exact command echo and copied/discarded transport result. |

The schedule uses the existing 41-word signed-int64 layout with production
version 2. A request identifies its epoch, generation, cycle, step, monotonic
target and cutoff. Version-1 schedules belong to the older provider path and
are not accepted as production schedules. Semaphore notifications mean
“reread the current coherent record”; they are not a lossless count of slots.
All scheduling timestamps are RTC `CLOCK_MONOTONIC` nanoseconds.

A command record contains 13 signed-int64 words: protocol version, schedule
epoch/generation, cycle/step, target/cutoff, source payload counter, ready time,
producer session/sequence/PID and source inode. Its acknowledgement echoes
those words and appends a transport status. Exact identities prevent stale
acknowledgements from releasing a newer command.

There is one outstanding source publication. After publishing its pixels and
ready record, the producer leaves that payload unchanged until synco snapshots
or discards it and acknowledges the exact command. This acknowledgement frees
the transport buffer; it does not wait for dmcomb, SDK return or physical DM
completion. The producer may calculate its next command in private memory
while waiting. Invalid/expired schedules stop timed submissions; restart and
generation changes invalidate old requests.

A normal producer process restart can reuse the same persistent source and
sidecar files. Replacing a source/sidecar backing file fails closed for that
mapping and retains its preceding destination contribution until operator
reacquisition. Incompatible initial image geometry or record layout can prevent
resource acquisition. The protocol does not provide live geometry migration.

## Configuration

Merge the [production preset](../dmtweeter_synco_production.conf) into the working
instance configuration. Preserve
the actual primary/combined names, observer CPU placement and controller
settings. Disable finite zero-test and read-only simulation modes explicitly
if those sections remain in the file.

```ini
[production]
enabled=true
readyLeadUs=50
step0LeadUs=20

[zeroTest]
enabled=false

[simulation]
enabled=false
readerReleaseReady=false
pipelineBackend=false

[scheduler]
maxSubframes=4
maxCadenceHz=8000

[streams]
destinations=dm01disp07,dm01disp08
timing=dmtweeter_synco_timing

[readerReleaseMonitor]
shmimName=dmtweeter_read_done
threadPrio=30
cpuset=spare57
```

The input-ready and passive-copy leads are preparation budgets, independent of
the publication target. Measure complete snapshot/copy work for the configured
geometry and number of mappings when qualifying a new setup.

For managed dmSpeckle, configure its DM channel as the mapped auxiliary source
and set the shared timing stream:

```ini
[syncopation]
timingStream=dmtweeter_synco_timing
writeLeadUs=50
```

The existing dmSpeckle channel/modulation configuration selects its source and
waveform. Managed mode does not independently trigger a main channel. Its
target-time waveform calculation remains in dmSpeckle; schedule validation,
buffer ownership and tagged publication are handled by the CRTP helper.
An empty timing-stream setting preserves legacy dmSpeckle operation.

The [vibration example](../../dmSpeckle/dmtweeter_vibration.conf) selects
`dm01disp07_aux`, a 200 Hz quadrature tip/tilt waveform and a separate producer
cpuset. Use small, operator-approved amplitudes when qualifying changing inputs.
Turning modulation off retains the contribution. A managed zero request queues
a tagged zero command for a future valid slot; it does not write the source
around an outstanding transport commitment.

## Adding another producer application

Derive from `dev::dmSyncopationProducer<yourApp>`, provide its friend declaration
and alias the base as `dmSyncopationProducerT`. Use
`DMSYNCOPATIONPRODUCER_SETUP_CONFIG(config)` in `setupConfig()`,
`DMSYNCOPATIONPRODUCER_LOAD_CONFIG(config)` in an int-returning configuration
loader, and `DMSYNCOPATIONPRODUCER_APP_SHUTDOWN` after the producer thread joins.
The application owns its writable source `IMAGE` and private command storage.

In that one publishing thread:

```cpp
synco::Request request;
if( dmSyncopationProducerT::syncopationPoll( request, source ) == 1 )
{
    // Compute the full contribution for request.targetNs into private pixels.
    dmSyncopationProducerT::syncopationPublish( pixels, request, source );
}
```

`syncopationPoll()` returns 0 while waiting and -1 for unavailable or
incompatible transport. Sleep between unsuccessful polls; this helper does not
create an application thread. `syncopationPublish()` waits to the configured
lead, revalidates the request, and returns whether a complete tagged payload
was submitted. It does not promise that a subsequent shared safety check will
allow the DM trigger. Do not modify the managed source through another app
callback or an independent zero/trigger path. Generate the next image in
private storage while a source commitment remains outstanding.

The read-only `production` INDI property reports passive copies/skips and
`accepted0`, `missing0`, `rejected0`, and corresponding indices in configured
destination order. Accepted counts mean private snapshots; they are separate
from destination-copy completion and DM delivery. Existing shared scheduler
and controller telemetry remains the evidence for actual publications,
acknowledgements, reader clearance and backend execution.

In production mode, `settings` exposes only `steps` and `maxCadenceHz`, editable
while disabled. The older keepout/guard/step-0 settings belong to the legacy
scheduler; production uses exact reader-release readiness and its configured
ready/passive-copy leads. The standard timing/health properties describe actual
production state, rate and publications, rather than a hypothetical shadow rate.

All real instrument startup, enable/disable actions and nonzero qualification
are performed by the operator. The zero trials establish transport capacity;
changing patterned commands and the selected multi-producer configuration
still require instrument verification.

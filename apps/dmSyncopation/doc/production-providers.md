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

Rebuild/install `libMagAOX`, the DM controller, `dmSyncopation`, `dmSpeckle` and
`logdump` from this revision before deployment. On the instrument, `teldump`
is an alias for `logdump --dir=/opt/MagAOX/telem --ext=.bintel`, not a separate
reader to build. Restart the affected applications through the normal operator
procedure after installing their instance configurations.

The tested 8 kHz setup uses the BMC controller's full
[preparation / SDK / post pipeline](../../bmcCtrl/doc/sdk-worker.md#full-preparation--sdk--post-pipeline),
with distinct isolated cores for the monitor, SDK worker, post worker, synco
and release observer. Its controller configuration includes:

```ini
[sdk]
workerEnabled=true
threadPrio=55
cpuset=spare56
pipelineEnabled=true
pipelineThreadPrio=30
pipelineCpuset=spare31
pipelineResultStream=dmtweeter_sdk_done

[dm]
completionStream=dmtweeter_done
readerReleaseStream=dmtweeter_read_done
```

These cpuset names are assignments in the tested RTC configuration, not portable
defaults. Select available dedicated cores and verify actual thread scheduling
and affinity on another host. Configure synco's primary/combined streams for
the real command path (`shmimMonitor.shmimName=dm01disp03` and
`streams.combined=dm01disp` in this example). dmcomb must react to the auxiliary
counter publications as well as the main writer. The tested configuration is
mode 4 / DELAY with a 5 us delay; mode 3 waiting only on the primary stream will
not provide the auxiliary combinations. RTC latency tuning was On in the
instrument timing tests. Requalify rates on other hardware or scheduling setups.

Merge the [production preset](../dmtweeter_synco_production.conf) into the working
instance configuration. Preserve
the actual primary/combined names, observer CPU placement and controller
settings. Disable finite zero-test and read-only simulation modes explicitly
if those sections remain in the file.

```ini
[production]
enabled=true
readyLeadUs=70
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
destinations=dm01disp07,dm01disp08,dm01disp09,dm01disp10
timing=dmtweeter_synco_timing

[readerReleaseMonitor]
shmimName=dmtweeter_read_done
threadPrio=30
cpuset=spare57
```

The input-ready and passive-copy leads are preparation budgets, independent of
the publication target. The tested four-channel setup uses 70/20 us, leaving
50 us between the producer cutoff and passive-copy completion cutoff. The
compiled input-ready default is 50 us; its 30 us preparation window was too
short for this mapping. Measure complete snapshot/copy work for the configured
geometry and number of mappings when qualifying a new setup.

For managed dmSpeckle, configure its DM channel as the mapped auxiliary source
and set the shared timing stream:

```ini
[modulator]
opMode=sparkle
cpuset=tweeterSpeck

[dm]
channelName=dm01disp09_aux
trigger=off
frequency=8000
dwell=1
amp=0.01

[syncopation]
timingStream=dmtweeter_synco_timing
writeLeadUs=50
```

The existing dmSpeckle channel/modulation configuration selects its source and
waveform. Managed mode does not independently trigger a main channel. Its
target-time waveform calculation remains in dmSpeckle; schedule validation,
buffer ownership and tagged publication are handled by the CRTP helper.
An empty timing-stream setting preserves legacy dmSpeckle operation. There is
no `opMode=managed` setting: use `sparkle`, `arbcube` or `tiptilt` with a nonempty
timing stream. The source must end in `_aux` and its corresponding destination
must appear in synco's mapping list.

For sparkle and arbitrary cubes, `frequency/dwell` is the plane-advance rate,
independent of synco's slot rate. At 8000/dwell 1, a four-plane sparkle advances
every nominal 125 us slot and repeats its full pattern at 2 kHz. At 2000/dwell 1,
it typically repeats each plane across four 8 kHz slots. In `tiptilt` mode,
`frequency` instead sets the physical vibration frequency of each axis.

Start synco before the producer so the shared sources and protocol records
exist. Both start disabled. After verifying the configuration and resource
readbacks, the operator enables continuous scheduling and then modulation:

```bash
setINDI -s dmtweeter-synco.enable.toggle=On
setINDI -s tweeterSpeck.modulating.toggle=On
```

The [vibration example](../../dmSpeckle/dmtweeter_vibration.conf) selects
`dm01disp07_aux`, a 200 Hz quadrature tip/tilt waveform and a separate producer
cpuset. Use small, operator-approved amplitudes when qualifying changing inputs.
Turning modulation off retains the contribution. A managed zero request queues
a tagged zero command for a future valid slot; it does not write the source
around an outstanding transport commitment.

To remove that producer's contribution, stop modulation and request zero while
leaving synco enabled so it can accept and copy the queued zero:

```bash
setINDI -s tweeterSpeck.modulating.toggle=Off
setINDI -s tweeterSpeck.zero.request=On
```

Confirm that the numbered destination contains the zero contribution before
disabling synco or stopping either application. A "zero submitted" log confirms
transport submission, not destination application. Stopping either application
alone does not remove the last copied contribution.

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

## Capture the running production setup

Keep synco enabled and tweeterSpeck modulating with the settings being tested.
Use this read-only block from anywhere inside the MagAOX checkout. It captures
120 seconds of telemetry after the before-snapshots finish; the bounded
before/after queries add to the total wall time. It saves the current synco and
producer configuration, including preparation leads absent from INDI readbacks.

The finite `capture-zero-aux-test.sh` helper controls a separate zero trial and
must not be used for this running production capture.

```bash
(
    prod_capture_dir="$HOME/tmp/synco-production-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$prod_capture_dir" || exit 1
    git rev-parse HEAD > "$prod_capture_dir/git.txt" || exit 1
    prod_devices=(dmtweeter-synco dmtweeter camwfs tweeterSpeck)

    prod_snapshot()
    {
        local prod_phase=$1 prod_device prod_status
        date -u +%FT%T.%NZ > "$prod_capture_dir/indi-$prod_phase-start.utc"
        for prod_device in "${prod_devices[@]}" holoop sysMonRTC.set_latency.toggle
        do
            if timeout -k 2s 8s getINDI "$prod_device" \
                > "$prod_capture_dir/$prod_device-indi-$prod_phase.txt" \
                2> "$prod_capture_dir/$prod_device-indi-$prod_phase.err"
            then
                prod_status=0
            else
                prod_status=$?
            fi
            printf '%s %s status=%s\n' "$prod_device" "$prod_phase" "$prod_status" \
                >> "$prod_capture_dir/status.txt"
        done
        date -u +%FT%T.%NZ > "$prod_capture_dir/indi-$prod_phase-end.utc"
    }

    for prod_config in dmtweeter-synco tweeterSpeck
    do
        if [[ -r "/opt/MagAOX/config/$prod_config.conf" ]]
        then
            cp -- "/opt/MagAOX/config/$prod_config.conf" "$prod_capture_dir/" || exit 1
        fi
    done

    prod_snapshot before
    prod_capture_pids=()
    date -u +%FT%T.%NZ > "$prod_capture_dir/collection-start.utc"
    for prod_device in "${prod_devices[@]}"
    do
        timeout -k 5s -s INT 120s stdbuf -oL -eL logdump \
            --dir=/opt/MagAOX/telem --ext=.bintel -f "$prod_device" \
            > "$prod_capture_dir/$prod_device.txt" 2>&1 &
        prod_capture_pids+=( "$!" )
    done

    for prod_pid in "${prod_capture_pids[@]}"
    do
        if wait "$prod_pid"
        then
            prod_status=0
        else
            prod_status=$?
        fi
        printf 'collector pid=%s status=%s\n' "$prod_pid" "$prod_status" \
            >> "$prod_capture_dir/status.txt"
    done
    date -u +%FT%T.%NZ > "$prod_capture_dir/collection-end.utc"
    prod_snapshot after
    printf '%s\n' "$prod_capture_dir"
)
```

The block waits only for its own four collector PIDs. A collector status of
124 is the expected timeout at the end of collection; 137 indicates forced
termination after the additional five-second grace. Nonzero INDI-query status
indicates a failed or incomplete snapshot. Keep `status.txt` and error files
with the capture, and share the printed directory path for analysis.

All real instrument startup, enable/disable actions and nonzero qualification
are performed by the operator. The zero trials establish transport capacity;
changing patterned commands and the selected multi-producer configuration
still require instrument verification.

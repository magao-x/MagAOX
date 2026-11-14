# Offline ADC/WFS timing audit

These standard-library Python tools read a completed native observation journal.
They never load ImageStreamIO, connect to INDI, start hardware/services, train a
model, generate corrections or publish a DM command. No application rebuild or
installation is needed. Input CSV/summary/configuration files remain unchanged;
reports go to stdout. DM transport is explicitly deferred by the `adc-wfs`
verifier profile, while the default `full` profile retains its activity check.

The operator's first native run reported 480,018 ADC and 120,000 modal/camera
samples over 60.000099736 s, with zero reported acquisition/journal losses and
one inactive DM observation. The actual CSV is not available on this development
workstation. Those counts are operator evidence, not a locally reproduced phase
or synchronization result. Run the tools on that existing recording first.

## Source review: intended synchronization versus available code

The intended hardware relationship is a continuous 8-kHz acquisition lattice
with every fourth readout at a defined phase of the 2-kHz WFS. For a stable source
incarnation, associate counter-labelled samples, not groups of four array rows:

    a[n0 + 4(k-k0)] - P[k]

Here `a` is an ADC acquisition marker and `P` is a verified camera publication
marker mapped onto the observer's monotonic clock. This is publication-relative
phase characterization, not an exposure timestamp or a causal feature selection.
The absolute counter residue is not assumed from attachment order.

Inspected sources and their exact checkpoint identities:

- Current branch `apps/mcp3208Ctrl/mcp3208Ctrl.hpp:472-523,1424-1696`:
  semaphore-wakeup period smoothing, signed post-delay, cadence budgeting and
  delay-error feedback. Its modeled delay is
  `D = (T/2 - (dtTransfer + wfsProcess + dtF + wfsRead) + postDelay) mod T`.
  The reference is the software semaphore receipt. Default latency terms sum to
  340.6 us; at T=500 us and postDelay=0, D=409.4 us. Projecting that phase onto a
  125-us ADC lattice gives 34.4 us modulo 125 us. Neither value is a measured
  camera-publication offset or an exposure boundary. Measured wakeup jitter,
  service time, delay caps and clock mapping remain additional terms.
- `parkertjohnson1/ad7689Ctrl`, commit `0ae4dcb8`,
  `apps/ad7689Ctrl/README.md:121-128,157-161` and
  `ad7689Ctrl.hpp:1696-1710,1905-2120`: software scan-start timestamp, sequential
  channel conversions, one complete scan per semaphore wake. A 2-kHz semaphore
  therefore does not itself produce 8-kHz samples. This revision does not verify
  the intended continuous hardware 4:1 synchronizer. The actual hardware
  revision/configuration must be supplied for the later hardware test. The
  audit supports the intended lattice without changing this producer. There is
  another source-level caveat: `ad7689Ctrl.hpp:2044` resets the commanded delay
  to its model target each synchronized cycle, while `:2142-2169` stores a feedback
  correction afterward. That carried correction appears overwritten at the next
  cycle. The MCP3208 path has the corresponding assignments at `:1549,1669-1696`.
  This is an inspection finding to check against the deployed revision; no
  producer fix or hardware-behavior claim is made here.
- Sibling accelSim worktree commit `9d4c5a99`,
  `apps/accelSim/accelSimModel.hpp:526-591` and `accelSim.hpp:697-833,873-909`:
  empty synchronization input selects an independent timer. Configured sync
  uses WFS-counter-labelled semaphore arrivals, an EMA period estimate and the
  same latency model; corrections slew a continuous sampling lattice with
  bounded headroom. Duplicate triggers are ignored and stale/replaced sources
  invalidate sync. A correct average rate alone cannot distinguish these modes.
- Current `libMagAOX/app/dev/frameGrabber.hpp:1016-1044`,
  `dev/edtCamera.hpp:549-560`, `apps/ocam2KCtrl/ocam2KCtrl.hpp:1592-1595,1716-1739`:
  publication uses REALTIME; camera acquisition metadata comes from EDT's timed
  acquisition API. The camera sync stream copies metadata; it adds no exposure
  bounds. Verify running versions and the SDK event/clock contract.

For the AD7689 branch, inspect without switching branches:

```sh
git show 0ae4dcb8:apps/ad7689Ctrl/README.md
git show 0ae4dcb8:apps/ad7689Ctrl/ad7689Ctrl.hpp
```

Delay/jitter discontinuities in the journal are labelled **candidates**, never
confirmed resynchronization actions. Confirmation needs producer trigger IDs,
receipt times, applied-delay/cap/lock information and configuration history.
Current one-timestamp scan records also cannot establish AD7689 channel aperture
skew. Real hardware requires its own calibration profile; `accelSim12` is not an
AD7689 calibration.

## What the audit computes

- Scoped transport: exact journal/summary counts, independent counter-gap checks,
  required ADC/modal/camera activity and observer-time coverage, approximately
  8/2/2-kHz rates, invalids/restarts/queue failures. DM remains recorded and
  deferred. Counter and timestamp arithmetic preserves full-width integers.
- Clock evidence: acquisition/publication declarations are distinct from verified
  same-host domains and event meanings. REALTIME/TAI clock brackets are mapped
  into monotonic intervals; unknown domains are not inferred from numerical
  resemblance. Mapping allows a conservative 1000-ppm offset drift between
  local beacons, at most 2-ms beacon age and a default 25-us half-width limit.
  Disjoint neighboring offset intervals beyond the drift allowance create clock-step segments;
  mappings crossing such steps are rejected. Neighboring same-epoch bracket
  bounds are conservatively combined rather than treating a sub-budget offset
  jump as zero uncertainty. This is a host-clock mapping
  allowance, not a hardware oscillator, exposure or phase-lock qualification.
- One-in-four phase: all four `counter % 4` populations; an initial nearest
  anchor per incarnation, then fixed counter-labelled pairing. The audit does
  not repeatedly pick a closer later sample to erase drift. Diagnostic residue
  selection uses only the initial second unless explicitly supplied. Outputs
  include offset, signed/absolute jitter percentiles, standard deviation/RMS,
  detrended jitter, drift in ppm, missing pairs, counter-stride failures and
  bounded examples of possible phase discontinuities. Smaller consecutive phase
  changes above `max(T/80, 2 * combined adjacent mapping uncertainty)` are counted
  as adjustments-or-jitter, to expose bounded slew without calling it a confirmed
  resynchronization. Large-step detection uses T/8 instead. No detector can
  identify all actual controller actions without producer telemetry. Drift is not fitted
  across source/clock incarnations. An expected publication phase must have
  independent evidence; the 409.4-us wakeup model is not substituted for it.
- A separate observer-receipt phase: always available when timestamps suffice,
  but explicitly biased by polling, ordering and batching. It cannot certify
  acquisition phase, reconstruction latency or hardware synchronization.
- Modal-camera association candidates: exact `cnt1 -> camera cnt0` and exact
  inherited-acquisition-timestamp matches. Reused/ambiguous/missing matches are
  counted. A frozen input-pixel/origin contract is needed to qualify either;
  equality or nearest timestamps alone are insufficient. Restarts are not
  silently joined by matching old counter values.
- Causal ADC availability at each recorded modal observer deadline: both mapped
  acquisition/publication upper bounds and actual frontend delivery must be at
  or before that deadline. Earlier-acquired but late-delivered records are
  excluded until available. Reports include ages, delivery delay, no-input/stale
  cases, incomplete 500-us diagnostic windows, sample counts and bounded example
  counter IDs. A window requires a left anchor, consecutive counters in one
  incarnation and no hold longer than 250 us. These conservative diagnostic
  limits are not a new DDSPC feature algorithm. They produce no acceleration
  feature vectors, qualified WFS records or training inputs. Missing/uncertain
  records are not padded; exactly four records per window is not required.

The main deadline here is **observer availability**, not exposure end, an AO
processing callback deadline or a synco input cutoff. Native code polls ADC,
then DM/camera/modal in a fixed order. Those receipt times cannot prove the
physical sensor phase or the true optical/control latency. A second
`causal_publication_cutoff` report uses the mapped modal-publication lower bound
as a conservative earlier cutoff. It never calls that marker the actual WFS
processing deadline. A true producer/callback deadline needs separate evidence.
Offline clock reconstruction may use brackets on both sides of a record; this
is an evidence audit, not a replay of an online clock estimator. Data admission
still requires actual recorded frontend availability before the chosen cutoff.

## Future dmtweeter-synco interface (no implementation or activation)

Inspected sibling worktree `9d4c5a99`:

- `apps/dmSyncopation/dmSyncopationProduction.hpp:89-116,125-206,258-286`;
- `libMagAOX/app/dmSyncopationTiming.hpp:21-92,229-267`;
- `libMagAOX/app/dev/dmSyncopationProducer.hpp:217-232,393-545`;
- `apps/dmSyncopation/doc/production-providers.md`.

The production v2 scheduler uses genuine **main DM command publication** as its
cycle origin. Four slots at 2 kHz have 125-us spacing. Displays may label the
next cycle 1.0/1.25/1.5/1.75 frames; relative to its main command those phases are
0/.25/.5/.75. The advertised step-zero request is for the next main target;
active steps 1..3 are current-cycle quarters. Neither label defines WFS exposure
or physical application time.

At the user-provided approximate 110-us write service, nominal slack is only
15 us. This does not include/qualify combined-publication jitter, reader
clearance, copying, clock uncertainty, backend tails, preemption or an early
main. The tool reports this arithmetic as an assumption, never feasibility.

Later DDSPC output integration must use the shared
`dev::dmSyncopationProducer` helper (`syncopationPoll` / `syncopationPublish`),
not write numbered channels or the combined image. Its request includes frozen
epoch/generation/cycle/step, monotonic target and **input-ready cutoff**. A source
must be complete before that cutoff, often substantially before the target.
The compiled ready lead is 50 us; the inspected four-channel preset uses 70 us
with a separate 20-us passive-copy lead and 30-us copy budget. These are source
configuration facts, not bounds qualified for DDSPC. The helper can require an
additional producer write lead, so calculation must finish earlier still.

The provider owns only its assigned `_aux` image and `_ready` sidecar; it keeps
the payload immutable until an exact `_ack`. ACK means snapshot/discard and
buffer release, not dmcomb/SDK/physical application. Expired/restarted schedules
invalidate requests; late commands must not migrate into later slots. Synco may
retain the preceding contribution after a missed input or producer stop, so a
future shutdown/zero policy needs explicit design and authorization. The current
2-kHz predictor does not automatically supply validated quarter-frame forecasts.
A two-mode DDSPC forecast also needs a verified modal-to-actuator/units mapping
before producing a tweeter contribution. None of this transport is imported,
created, enabled or exercised by this audit.

## Run on the existing exao2 recording (operator only)

Make the reviewed Python files available in your user checkout or a private
analysis directory, keeping `native_timing_audit.py` beside
`verify_native_observation.py`. Do not replace any installed executable. No
native application, INDI broker or producer needs to run for this analysis.

```sh
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export DDSPC_TOOLS="$HOME/source/MagAOX-DDSPC/apps/loPredCtrlAcc/tests"
export DDSPC_OBS="$HOME/ddspc-native-observe.9UlnxD/observations"
export DDSPC_AUDIT_OUT="$(mktemp -d "$HOME/ddspc-timing-audit.XXXXXX")"

python3 "$DDSPC_TOOLS/verify_native_observation.py" "$DDSPC_OBS" --profile adc-wfs \
  | tee "$DDSPC_AUDIT_OUT/acquisition.json"
python3 "$DDSPC_TOOLS/native_timing_audit.py" "$DDSPC_OBS" \
  | tee "$DDSPC_AUDIT_OUT/timing-unqualified.json"
```

On the reported data, expect the scoped acquisition pass with `dm_transport`
`deferred`. Unverified clocks/association leave `offline_prediction_timing_ready`
false. `observer_receipt_phase` is descriptive only. The actual acquisition phase
may be unavailable until an explicit clock manifest is supplied; this is expected.

For a numeric hypothesis using the inspected simulator/framegrabber clock
candidates, copy the example into the **new analysis directory**:

```sh
cp "$HOME/source/MagAOX-DDSPC/apps/loPredCtrlAcc/examples/native-timing-provenance.json.example" \
  "$DDSPC_AUDIT_OUT/provenance.json"
python3 "$DDSPC_TOOLS/native_timing_audit.py" "$DDSPC_OBS" \
  --provenance "$DDSPC_AUDIT_OUT/provenance.json" \
  | tee "$DDSPC_AUDIT_OUT/timing-hypothesis.json"
```

All example `verified` flags are false. Edit only this private analysis manifest
when source/build/clock/event evidence supports it. Identify the actual modal
publisher and ensure frame tags accompany the same pixels. Do not select a clock
or association because it gives a favorable phase plot. Leave expected phase
null until the reference and offset have independent evidence. Replace latency
terms with recorded configuration, without modifying producer configuration.
An unknown trigger-to-publication relation blocks an absolute lock claim.

For a strict diagnostic timing gate:

```sh
DDSPC_TIMING_STATUS=0
python3 "$DDSPC_TOOLS/native_timing_audit.py" "$DDSPC_OBS" \
  --provenance "$DDSPC_AUDIT_OUT/provenance.json" --require-timing-ready \
  > "$DDSPC_AUDIT_OUT/timing-gate.json" || DDSPC_TIMING_STATUS=$?
printf 'Timing audit exit status: %s\n' "$DDSPC_TIMING_STATUS"
```

Exit 0 means transport passed and, when requested, the diagnostic timing gate
passed. Exit 1 means transport failed; exit 2 with `--require-timing-ready` means
unresolved timing evidence. Inspect `timing_review_blockers` in every case.
Malformed evidence exits with an error. Without the strict flag, an acquisition
pass with unqualified timing deliberately exits 0 and reports those blockers.
No status grants training or DM permission.

## Criteria for the next offline prediction experiment

1. Full-duration ADC/modal/camera transport passes; DM qualification stays deferred.
2. All timestamps used for alignment have source/domain/event evidence on the
   same host, complete mapping coverage and <=25-us mapping half-width by the
   chosen diagnostic budget. No bridged clock steps, stale mappings or order
   violations. The producer timestamp/aperture error is an additional bound.
3. Modal-camera origins are unique and evidenced. Unknown or missing provenance
   is a blocker, not a guessed nearest-frame match.
4. The selected incarnation-local one-in-four phase matches an independently
   established publication-reference target within the configured tolerance,
   with no missing counter-labelled pairs or unexplained discontinuities.
   Drift/jitter, all four residues and phase-selection calibration are reported.
5. Actual frontend availability supports fresh causal inputs at the recorded
   modal deadlines; no future/late records are admitted and diagnostic windows
   have complete bounded coverage. Review startup/tail exclusions separately;
   the automated gate currently fails incomplete coverage conservatively.

A passing `offline_prediction_timing_ready` only supports designing the next
**offline** measurement/prediction experiment. `recorded_ddspc_inputs_qualified`
remains false: replay through the actual qualified DDSPC path still needs valid
WFS exposure timing and independently confirmed applied-command histories, or a
separately justified synthetic-input experiment. An inactive command stream does
not prove zero applied input. No training or closed-loop correction is enabled.

The existing simulator recording can characterize its cadence, publication phase
hypotheses, delivery and evidence gaps. It cannot qualify the real synchronized
ADC, its resynchronization controller, channel aperture skew, optical coupling,
physical DM application or synco scheduling. A later explicitly authorized real
sensor observation needs its actual synchronization revision, trigger IDs/times,
delay-adjustment/lock/cap telemetry, calibration and clock provenance. A later
active-command test must establish genuine main/synco/reader timing independently.
If real ADC acquisition occurs on another host (for example a Raspberry Pi), its
clock must not be declared same-host merely because REALTIME values look close.
Cross-host clock mapping/uncertainty and bridge timestamp semantics are separate
required evidence; this first audit deliberately does not invent that mapping.

## Local validation

Run on a personal development workstation:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline check-native-timing
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-native-observation
```

The Python target passes 33 cases (22 new timing cases plus existing build and
artifact tests). The unchanged guarded application/native-observation tests
pass 30 C++ cases / 10,438 assertions. Tests cover missing/inactive sources,
counter gaps/restarts, clock mismatch/steps/uncertainty, jitter and 200-ppm phase
drift, full-cycle skips, gradual corrections, delayed/stale/uncertain samples,
ambiguous/reused origin tags, 64-bit IDs above 2^53, read-only CLI behavior and
permanently inhibited training/output. Only Python, test targets and documents
changed; no additional native dependency or application build is required.

A separate 60-second synthetic journal exercised 480,016 ADC, 120,000 modal,
120,000 camera and one DM sample (720,017 rows, 84.3 MB). With explicitly synthetic
clock/origin/phase evidence, it recovered 30-us phase and zero drift, admitted no
future samples at either deadline reference and preserved input SHA256 hashes.
The audit took 17.6 s with about 210 MiB peak process RSS on the development Mac.
This is an offline file-processing test, not the exao2 recording, hardware
synchronization, full 165 MB-format throughput or a real-time performance result.

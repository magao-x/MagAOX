# Offline ADC/WFS timing audit

2026-11-13. Workstation ASTR-LT-428.local, branch
parkertjohnson1/accel_ddspc, start 0d867399. Observation-only Python tools and
tests; no C++/configuration/producer changes, deployment, services, Git staging,
commit or push. The operator's 60-second recording is not present locally.

Read-only sources: MCP3208 synchronization in this checkpoint; AD7689 branch
parkertjohnson1/ad7689Ctrl at 0ae4dcb8; accelSim and dmSyncopation in the sibling
worktree at 9d4c5a99. AD7689 currently documents one scan per semaphore wake:
its source does not establish continuous 4:1 hardware synchronization. The audit
must test the intended four-sample lattice without presenting it as verified
hardware behavior. Preserve the latency-model reference to semaphore receipt,
not camera exposure or publication. Distinguish default/assumed model parameters
from measured phase. Counter modulo four is incarnation-local; lost samples do
not renumber the remaining array into a new phase. Freeze diagnostic residue
selection from an initial calibration interval and report each residue.

Reuse the committed CSV fields for clock brackets and observer availability.
No qualified TimedWfsRecord or applied commands are created. Clock declarations
alone are hypotheses; an explicit source/clock evidence manifest is needed for
a timing gate. Candidate resynchronizations remain candidates without producer
telemetry. Publication-based phase is never relabeled exposure phase.

Synco production version 2 uses main-command publication as its phase origin,
not WFS exposure/publication. Four steps at 2 kHz imply 125 us spacing; 110 us
service leaves 15 us nominal slack before other costs. Provider input cutoffs
precede targets (50 us compiled ready lead, 70 us in the inspected preset),
with separate copy/read-release/clock guards. Eventual integration must use the
shared producer helper, immutable slot identities and exact acknowledgements.
No synco/native shared-memory imports or output transport belong in these tools.

Implement scoped ADC/WFS verification, offline clock/phase/association/causality
analysis, deterministic adverse-condition tests and reproducible operator-only
commands. Validate current tools, new tests, finite 60-second synthetic journals
and guard regressions. Record limits and results in NATIVE_TIMING_AUDIT.md.

Implementation and local validation:

- `tests/verify_native_observation.py`: default full behavior retained; explicit
  ADC/WFS-only activity/coverage/cadence and independently counted journal gaps;
  DM activity deferred while preserving artifact-integrity checks.
- `tests/native_timing_audit.py`: compact integer journal columns, bracketed
  host-clock hypotheses/attestations and step boundaries, four counter residues,
  frozen counter-labelled phase pairing, circular/unwrapped offsets, drift,
  jitter and bounded slew/step candidates; exact association candidates;
  ADC availability at observer and conservative publication-marker cutoffs.
  No data are relabeled as exposure/physical-input evidence. Future synco
  timing math is explicitly hypothetical and creates no transport.
- New `tests/native_timing_audit_test.py`, `examples/native-timing-provenance.json.example`,
  and `NATIVE_TIMING_AUDIT.md`; updated `tests/Makefile.offline` and
  `NATIVE_SHADOW_TESTING.md` (including obsolete Eigen3.4-only gate removal).

Readiness remains fail-closed for unknown clocks/events/origin/expected phase,
incomplete mappings/coverage, restarts, clock steps and counter discontinuities.
Whole-WFS-cycle skips cannot hide behind wrapped phase: the test drove a fix to
use the counter-anchored residual directly. Short-offset clock changes cannot
hide inside the separate mapping acceptance budget. Late/uncertain ADC samples
are deferred until causal. All learning/DM/exposure/application permissions stay
false even on a passing diagnostic timing gate.

Source caveat: AD7689 0ae4dcb8 resets m_synchroDelay to its target at line 2044,
after storing feedback at 2142-2169 in the previous cycle. MCP3208 has the same
pattern. The carried correction appears overwritten; verify the deployed
revision before claiming persistent hardware PLL behavior. No producer changed.
Remote-host acquisition needs independently qualified cross-host mapping; the
new tool deliberately rejects a falsely same-host clock claim.

`make -f tests/Makefile.offline check-native-timing`: 33 Python cases passed,
including 22 new timing cases. Existing guarded native-observation/application
suite: 30 C++ cases / 10438 assertions passed. No C++ source or algorithm changed;
no new native Linux/ASan/TSan qualification is claimed for Python-only changes.

A separate synthetic 60-second 8/2/2-kHz journal (720017 sample rows, 84,277,761
bytes; not the instrument recording) passed acquisition and evidenced synthetic
timing checks, recovered 30-us phase/zero drift, admitted zero future samples at
both deadline references, and retained input SHA256 hashes. Final audit time
17.61 s, peak process RSS 219,922,432 bytes on this personal Mac. This measures
unpaced offline file processing, not real-time acquisition or synco readiness.
Logs/reports: /private/tmp/ddspc-timing-audit-rh00cc4u/results/.

Input synthetic CSV SHA256:
4321388a5edf0c98ad70265c4fd74748d171c43a40c51d2411bcd1584fd9b947.
Actual exao2 CSV/configuration remain unavailable locally; the guide targets
that existing 60-second run first and leaves its phase values unclaimed. No
native app, INDI/hardware connection, deployment, install, service, Git staging,
commit or push was performed. Original ADC calibration/controller/producer and
tracked numerical results are unchanged. All changes remain for user review.

## 2026-11-14: mapping and alternating-phase diagnosis

Start checkpoint 592d124f, same personal workstation and feature branch.
The latest exao2 audit JSON, exact manifest and recorded CSV are still absent
locally. Requested their local location; no instrument/remote connection was
made. The observed 105-us alternation cannot be attributed to a measured cause
without those artifacts. Preserve that distinction in the operator report.

Only ordinary-file Python analysis, tests and documentation are changed:

- Count every rejected ADC/camera/modal acquisition/publication conversion by
  reason; retain bounded counter-labelled examples and observer clock coverage.
  Keep all original clock-domain, 25-us uncertainty and 2-ms age admission rules.
- Expose fixed-pair anchors, raw/mapped timestamp increments, uncertainty sums
  and observer receipt increments. Compare camera acquisition/publication
  markers on the same rows independently of any ADC pairing.
- Reproduce two distinct synthetic 105-us effects: stable acquisition with
  alternating 395/605-us publication intervals; and a separate nearest-any-ADC
  heuristic choosing strides 3/5 when its reference changes only 20 us. The
  current fixed-residue audit retains stride 4 and reports 20 us in the latter
  case. Its stride rule is imposed, not independent evidence of hardware lock.
- Fix the demonstrated offline period-estimator defect: alternating publication
  intervals can bias the adjacent median to 605 us, and missing alternate clock
  mappings can eliminate adjacent pairs. Use time/counter spans inside mapped
  incarnations; retain the adjacent median as a diagnostic. No source timestamp,
  DDSPC algorithm or acquisition behavior changes.
- Explicitly report synchronization reference, actual WFS processing deadline
  and synco slot association as unqualified. Source inspection confirms semaphore
  receipt and main-DM publication are distinct references. Document real-reference
  phase/error and causal admission equations, missing telemetry, exact original-
  manifest rerun commands and future tagged synco cutoff requirements.

`check-native-timing`: 39 Python cases passed, including six added regressions.
The 60-second synthetic journal mapped all six timestamp fields, retained the
30-us synthetic phase with no future admissions, and preserved all three input
SHA256 hashes. Current audit elapsed 20.43 s; peak process RSS 229,310,464 bytes
on macOS. This is offline synthetic throughput, not the instrument result or a
real-time/scheduler qualification. Artifacts:
`/private/tmp/ddspc-timing-diagnosis-zv8_d5ci/`.

No C++/configuration/calibration/controller/producer or tracked numerical result
changed. No acquisition, INDI, hardware service, output transport, deployment,
installation, Git staging, commit or push was performed. Existing C++ guard
qualification is unchanged; Python-only changes do not claim new native Linux
or sanitizer qualification. Actual run attribution remains pending its artifacts.

## 2026-11-14: timestamp writer and recorded-run provenance review

Start 3f9109d1, same personal workstation and feature branch. The operator's
new diagnostics reject every sample before numerical conversion: ADC lacks
same-host provenance, camera/modal domains are unknown. The 55,427 REALTIME
beacons and 0.05-us median full bracket are operator-reported, not locally
reproduced. Actual recording/build/process artifacts remain absent locally;
requested saved writer/host/build evidence without contacting the instrument.

Reviewed local accelSim/camera/frameGrabber sources and pinned public MILK,
CACAO, ImageStreamIO and reference configuration sources. Public-source reads
and private temporary copies do not identify the installed instrument build.
The conventional configured modal producer is wfs2cmodeval-1, using
linalg.MVMmextrmodes on aol1_imWFS2. The reference helper passes NULL acquisition
time to ImageStreamIO: atime/writetime become output-update CLOCK_ISIO, default
TAI when available unless overridden. The separate CLOCK_MILK process trace is
not the metadata writer. Camera origin is not propagated into metadata cnt1;
trace counters are not automatically frozen input-pixel certificates.
nnReconstructor is a possible alternate writer of the same stream, with
REALTIME callback-arrival atime and frameGrabber publication time. Its existence
prevents identifying the recorded active writer from the stream name alone.

ADC sample-generation and camera/ADC frameGrabber publication use REALTIME in
the reviewed paths. EDT acquisition copies its DMA-buffer SDK stamp; default
host-system timing is documented, while active library/driver/mode need binding
to this interval. Reference RTC placement supports exao2 as intended host but
does not exclude a remote timestamp-preserving relay during the actual run.

Added NATIVE_TIMESTAMP_PROVENANCE.md and a schema-1 partial manifest with no
verified or same-host fields, plus a navigation link in NATIVE_TIMING_AUDIT.md.
No arbitrary TAI offset, producer attestation, exposure, association, phase-lock
or applied-command certificate is created. Existing REALTIME beacons can serve
qualified same-host REALTIME fields but not unknown TAI/cross-host relations.
The smallest next step is run-specific writer/library/host evidence, then
offline mapping of proven fields; ADC/camera diagnosis need not await DM/modal
qualification. Hardware 4:1 lock and future synco cutoff scheduling stay distinct.

Validation: 39 existing Python cases pass. The new partial JSON loads through
the actual audit on a private synthetic fixture, grants zero mappings, and
leaves every readiness/training/output/physical qualification false while
synthetic transport passes. No algorithm/test/runtime source was changed.
Whitespace checks pass; no C++ build or sanitizer rerun is claimed for these
documentation/JSON-only changes. Source ledger and copies are under
/private/tmp/ddspc-provenance-sources-jmlqg8lo/. No native application, services,
hardware, deployment, installation, staging, commit or push was performed.

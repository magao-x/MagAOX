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

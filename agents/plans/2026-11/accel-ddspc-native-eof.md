# Offline native-observation EOF validation

2026-11-14. Personal workstation ASTR-LT-428.local; feature branch
parkertjohnson1/accel_ddspc, starting commit 3f9109d1. Preserve the user's four
already staged timestamp-provenance files and index. No staging, commit or push.

Implement only ordinary-file tools. The native observation resides on exao2
and is unavailable locally; do not connect to retrieve it or claim its scores.
It contains simulated ADC without corresponding physical DM disturbance, so
forecast comparisons are a negative-control software experiment. Real AD7689
phase locking and eventual synco-slot/DM scheduling remain separate work.

Reuse the integer journal series/cadence helpers and ADC/WFS transport verifier.
Read native decoded ADC and selected modal values without reimplementing ADC
calibration. Preserve raw acquisition/publication and observer receipt/available
times. Use explicitly provisional frozen 4:1 counter anchors from past observer
availability, with no unverified clock conversion or physical ancestry claim.
Reject missing, invalid, late/stale groups and purge dependent histories at
source changes/counter gaps. Retain counter holes instead of renumbering rows.

Use H100, horizons1/2/4/8, purged 60/20/20 raw-row partitions and a common valid
origin mask. Fit bounded-batch standardized covariance EOF ridge models with
training-only transformations, validation-only ridge choice and frozen test
predictions. This separate predictor does not exercise production DDSPC/RLS,
applied-input pairing, readiness/fallback or control optimization. No fabricated
input certificates, commands, counterfactual DM residuals or physical timings.

Include causal delayed-ADC controls, independent training-statistic colored
sensors, timing controls and ordered held-out blocks. Treat their comparisons
as descriptive, not significance/physical-performance evidence. Test deliberate
causal sensor information, uncoupled noise, future-label invariance, source loss,
unverified clocks, exact large counters, singular fits and gapped PSD windows.
Export static plots/tables/frozen weights/predictions and fail-closed diagnostic
artifacts; record hashes, versions and unpaced timings separately from deadlines.

Implementation: new tests/offline_eof.py, tests/offline_eof_test.py,
tests/requirements-eof.txt and OFFLINE_NATIVE_EOF.md. Add only a dedicated
check-offline-eof target in tests/Makefile.offline, without adding numerical
dependencies to the existing standard-library native-timing target. No C++,
controller, acquisition, installed file, service or tracked result is changed.

The user committed the previously staged provenance work during this task;
the observed branch checkpoint advanced to 30502b64. No agent Git mutation was
performed, and those files remain unchanged by this implementation.

Validation: 24 new Python cases pass, including direct SVD/ridge equivalence,
unchanged WFS-only fit under ADC perturbation, actual combined forecast response
to ADC changes, zero/rank-deficient features and no held-out label/scaler leakage.
The 39 existing standard-library native timing cases also pass. CLI tests render
all plots, retain diagnostic failure artifacts, reject output collisions/input-
tree writes and preserve input hashes. Missing source atime stays missing and
unqualified without erasing finite observer-timed measurement payloads.

Two separate 60-second synthetic fixtures use 480018 ADC and 120000 modal/camera
rows each at nominal 8/2/2 kHz. H100/default horizons/controls yield
69892/21892/21892 common purged origins, full primary ranks200/200/400 and
99.99917% provisional group coverage (one startup group excluded). Independent
combined gains range from -0.4148% to -0.0074%; a deliberate causal ten-frame
sensor lead yields 95.6783%-98.0311% lower forecast MSE. These are synthetic
information tests, not actual native observation or physical control results.
Per-axis tables and reproducible fixture commands are in OFFLINE_NATIVE_EOF.md.
Private artifacts: /private/tmp/ddspc-offline-eof-2j0a8z05/, reviewed-* children.
Reviewed runs took 34.63/34.73 s reported wall time including plots/I/O,
8.50/8.47 s for association/all eleven EOF fits/evaluation, and 8.68/8.70 s
for loading/transport verification. This is unpaced Python3.13/NumPy2.3.5/
SciPy1.16.3/Matplotlib3.10.6 throughput, not real-time readiness. The current
analysis source SHA256 is 4fd8990515b312074ece2f9f02ad2a7a00f61d62f0e662ccc01803744285f6c8.
Final reviewed plots were inspected visually; ruff and whitespace checks pass.

No C++/native/controller/producer/calibration/operational configuration or tracked
numerical result changed. The new numerical target is separate from the existing
stdlib timing tests. No new C++/sanitizer/hardware/deadline qualification is
claimed, and no native executable/service, install, deployment, staging, commit
or push was performed. Actual exao2 data processing remains operator work.

# Shared ADC features and application shadow replay

2026-11-12, `parkertjohnson1/accel_ddspc`, personal workstation
`ASTR-LT-428.local`. The user authorized shared component extraction and an
opt-in application integration with offline replay, regressions, sanitizers and
2-kHz WFS / 8-kHz ADC timing tests. The user reports accelSim operational issues
are fixed; no producer, native application, server, service or hardware is run
by this task. No install, commit, push or deployment is performed.

## Scope and implementation record

- Move timestamp-window integration into `adc_features.hpp/.cpp`; preserve float
  accumulation/qualification and offer preallocated span-based processing.
- Move requested/effective selection, hysteresis and constraints into
  `shadow_policy.hpp/.cpp`; support arbitrary positive control dimensions and
  preallocated evidence/output, preserving the two-mode offline default.
- Keep the simulation's old names as thin aliases/adapters in
  `tests/offline_control.*`. Both simulation and app use the same algorithms.
- Add `ShadowPipeline` borrowing the application's actual `ShadowController`.
  ADC samples are decoded by actual `AdcCapture` before entering this path.
  WFS origins freeze actual shared aggregates; caller-provided WFS acceleration
  is discarded. Only historical records emitted by the actual causal pairing
  builder reach learner `process()`.
- Add ADC generation/segment metadata to frozen WFS and paired records so delayed
  confirmation cannot revive a previous source incarnation. Restart/gap changes
  assisted history without resetting the standard learner. Wrong clocks,
  malformed evidence, reversed time and incomplete windows fail closed.
- Add disabled-by-default `parameters.shadow_pipeline`. It requires shadow
  execution, explicit candidate bounds and a separately established input contract.
  Invalid opt-in loading also prevents direct startup. Native raw callbacks still
  discard unverified sessions; no live source adapter is invented.
- Add candidate/status seams and read-only pipeline/supervisor diagnostics. Existing
  diagnostic model selection and per-model training controls retain their meanings.
  Training is disabled initially. Shadow candidates never become AppliedInputRecords.
- Strengthen `DDSPC_SHADOW_ONLY` direct acquisition/publication entry points with
  unconditional compile-time rejection, preserving ordinary Legacy dispatch.
  Shutdown closes admission before input teardown and destroys the borrowing
  pipeline before the owning model pair.

No DDSPC identification/prediction/optimization equations change. Numerical core
CPPs and original `apps/loPredCtrl` files are retained; `shadow_controller.hpp`
only updates the old no-application-caller documentation. Causal pairing only
copies additional source provenance and retains its timing rules.

## Replay contract

`tests/shadow_replay.*` generates all records independently before callbacks; its
sink has no candidate getter or actuation interface. ADC publication/receipt,
WFS exposure/decision and closed applied-input intervals use one explicitly
synthetic clock/identity contract. The held inputs cover response-aligned WFS
exposures and their certificates arrive before the next decision, so current
prediction state has established historical actual inputs. Latent physical
forcing affects WFS only; decoder output supplies the acceleration features.

The actual application header is exercised with existing framework and
ImageStreamIO lifecycle/API doubles. It is not a second application implementation
or native binary launch. Different requests/slew limits must produce different
candidates but bit-identical learned models from the same confirmed-input data.
Unconfirmed driver-prepared records cannot train either learner. No policy output
is fed back into the recording generator or the confirmed-command seam.

## Validation and performance

Final command/result evidence is recorded in `apps/loPredCtrlAcc/SHADOW_PIPELINE.md`.
The application replay benchmark measures actual callback work under virtual
2-kHz / 8-kHz cadence, including synchronous identification and input pairing.
It separately verifies zero allocator requests for extracted in-place components
at 2, 3 and 10 control dimensions. Full pipeline allocation and timing evidence are
reported, not assumed to satisfy native real-time scheduling. The existing
asynchronous worker prototype remains a separate offline experiment.

Final strict and ASan/UBSan suites each pass 114 cases. Guarded application TSan
passes 21 cases; the actual ADC worker with the offline transport double passes
13 cases under TSan. The repeated 24-scenario / 72-trajectory fallback matrix is
byte-identical to its pre-extraction archive. Existing numerical algorithm sources
and all original `loPredCtrl` files retain their pre-task hashes.

At 10,000 replay frames, combined callback-work p99/max are 17.166/72.624 µs
for the small two-mode profile and 110.125/442.709 µs for the larger two-mode
example. Neither measured profile exceeds 500 µs. Separate ten-mode core timing
does exceed that period; no general synchronous real-time readiness is claimed.
Both shared in-place components show zero allocator requests; existing full
pairing/orchestration still allocates. New CSVs preserve all component results
without replacing earlier results. An initial cached core build mixed relative
and absolute dependency target paths; a fresh build resolves that ABI mismatch.

## Remaining live-integration gates

Native WFS snapshot/exposure mapping, ADC clock/calibration envelopes, independently
established actual applied-input certificates/modal projection, and source-silence
watchdog scheduling remain required. Input records are currently supplied by the
explicit adapter seam; no INDI configuration manufactures verified provenance.
The native raw WFS callback remains conservative and invalidates models.

Existing synchronous pairing/model work and allocation need assessment from the
reported timing data before promoting workers or claiming deadline guarantees.
Physical command ownership/handoff and safe operating-point policies remain
outside this shadow-only task. Synthetic bounds and ramp-to-zero diagnostics
must not become an instrument DM fallback policy. The 56-Hz issue remains a
separate offline identification/tuning/generalization investigation.

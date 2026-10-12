This work was performed by GPT-6 in response to the prompt: "Alright, please implement the proposed fixes and prepare everything for a pull request (dont do it yet, I will do another testrun today)."

This PR prepares the validated modal PO4AO application for MagAO-X, preserving
the implementation recorded at `b6d5ab65` and correcting the subsequent AGENTS.md
audit findings.

PO4AO publishes differential modal commands upstream of `mfilt`, supports
TensorRT and CPU linear-local inference, controls a configurable modal prefix,
and performs model loading and replay publication on background threads.
Startup/reset select proportional fallback until an explicit compatible reload.
Replay metadata, source discontinuity handling, and latency/model telemetry
support online training and experiment diagnostics.

The audit fixes reject old-generation models at activation, preserve background
model destruction, order ready/pending publication correctly, propagate monitor
errors, move helper definitions below declarations, and complete interface/test
documentation and repository formatting.

Validation: the local Catch2 suite passes 72 assertions in 13 cases with C++17
and release-style `-ffast-math` flags, and again with AddressSanitizer and
UndefinedBehaviorSanitizer. Clang-format 16.0.6 and whitespace checks pass.
The operator's full CUDA/TensorRT build and next instrument test are required
before opening/merging this PR.

Python training and runtime configuration remain separate repositories; this
PR requires their compatible version-2 runtime configuration and published
model contracts. No INDI driver list or CACAO configuration changes are included.
The engineering record is `apps/po4ao/PR_PREPARATION.md`.

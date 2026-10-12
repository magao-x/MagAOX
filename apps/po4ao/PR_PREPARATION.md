# PO4AO PR preparation

This work preserves the validated `rlandman/po4ao-controlled-modes` branch at
`b6d5ab65` and prepares `rlandman/po4ao-pr` from upstream `dev` at `02363bcf`.
The new baseline runtime files are byte-identical to the validated snapshot.
The upstream `picamCtrl` test registration is retained alongside PO4AO's entry.

## Engineering changes

- Validate the candidate generation again when the real-time thread takes a
  pending model. A reset can occur after the loader's original check.
- Send stale candidates to the background retirement slot. Leave candidates
  pending while that slot is occupied; never destroy a model in the callback.
- Publish the ready status before the candidate pointer. Publishing the status
  afterward could overwrite the idle status already set by activation.
- Use `SHMIMMONITOR_*` interface macros for configuration and app logic. Keep
  explicit startup/shutdown calls where an early return would skip worker
  cleanup, and propagate shutdown failures after cleanup.
- Move helper method definitions below their class declarations while keeping
  the application's header-only implementation.
- Complete declaration/test documentation, reconcile current behavior notes,
  and finish with a separate formatting-only commit.

## Validation

- Preserved runtime baseline: 55 assertions in 10 Catch2 cases passed locally
  under C++17, `-O2 -ffast-math -Wall -Wextra -Werror`.
- Runtime changes: 72 assertions in 13 Catch2 cases passed locally with the
  same release-style flags, including stale-candidate rejection, acceptance
  after reset, and preservation of a busy retirement slot.
- Moving all helper definitions below their class declarations preserved the
  same 72 passing assertions in 13 cases.
- Final formatter checks: pending.
- Full CUDA/TensorRT app build and physical test run: pending operator testing.

## Current repository instructions

Upstream `dev` now prohibits agent commands on instrument computers and INDI
tunnels. All implementation, formatting, and CPU tests are performed locally.
File headers omit `\author`, as required by current upstream instructions.
New commits use the namespaced feature branch and identify GPT-6 as co-author.
No PR is opened before the operator's next test run.

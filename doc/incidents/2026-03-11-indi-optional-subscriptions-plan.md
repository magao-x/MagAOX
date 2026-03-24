# Plan for INDI Subscription Retry Backoff (2026-03-11)

## Goal

Prevent persistent `getProperties` traffic when an application subscribes to INDI properties that are absent, delayed, or temporarily unavailable.

The immediate motivation is the ICC `logopt` case, but the fix should be general and reusable across all MagAO-X apps using `registerIndiPropertySet` / `REG_INDI_SETPROP`.

## Problem Statement

Current behavior retries every unresolved subscribed `set` property at application loop cadence:

1. an app registers interest in a remote property;
2. `MagAOXApp::sendGetPropertySetList()` keeps requesting unresolved properties;
3. retries happen every app loop until a matching `Def/Set` is seen;
4. if the property never exists, polling continues indefinitely.

This can generate unnecessary INDI traffic and can contribute to sustained `indiserver` CPU load.

## Proposed Approach

Introduce retry state and exponential backoff for all INDI `set` subscriptions.

Each monitored property should carry retry metadata so unresolved subscriptions are polled less and less often, up to a capped retry interval.

This preserves eventual self-healing when:

- a publishing app starts late;
- a publisher adds its properties after startup;
- a remote driver or `indiserver` restarts;
- a deployment changes and the property becomes available later.

The initial implementation should not distinguish between required and optional properties.

## Scope of Work

### 1. Extend the Set-Subscription Metadata

Update the internal callback/subscription record in [MagAOXApp.hpp](/home/jrmales/Source/MagAOX/libMagAOX/app/MagAOXApp.hpp) to include:

- `uint32_t retryCount`
- timestamp for last request
- next eligible retry time or current retry interval
- `bool missingLogged`

This will allow unresolved properties to be retried based on elapsed time instead of once per main-loop pass.

### 2. Implement Exponential Retry Backoff

Modify `sendGetPropertySetList()` so that unresolved properties retry with exponential backoff instead of every loop.

Initial intended behavior:

- retry at 1 s, 2 s, 4 s, 8 s, 16 s, 32 s, 60 s, then remain at 60 s;
- any later matching `Def/Set` still marks the property received immediately;
- a full refresh path should reset the backoff state and retry immediately.

### 3. Add Rate-Limited Logging for Missing Properties

When a property remains unresolved for a meaningful interval:

- emit one notice-level log indicating it is still unresolved;
- avoid repeated log spam;
- optionally emit a later recovery message if the property eventually appears.

This preserves observability without creating a new class of hard-to-debug silent failures.

### 4. Preserve Refresh / Restart Recovery

Ensure that any existing full-refresh path resets retry state so that delayed or restarted publishers are rediscovered promptly.

This includes the current behavior where the app infers a possible `indiserver` restart and re-requests monitored properties.

### 5. Audit Known Offenders and Validate Improvement

Perform a focused audit of other apps using `REG_INDI_SETPROP` or `registerIndiPropertySet` for advanced features that may be absent in some deployments.

This audit will be used to verify that backoff-first materially reduces persistent traffic even without introducing optional/required semantics.

## Implementation Order

1. Add metadata fields to the internal set-subscription record.
2. Refactor `sendGetPropertySetList()` to honor retry timing.
3. Reset retry state on refresh paths.
4. Add one-time or rate-limited unresolved-property logging.
5. Add or update tests.
6. Verify behavior against known problem cases such as `logopt` / `hogopt`.
7. Do a small follow-up audit for other obvious persistent subscribers.

## Testing Plan

### Unit / Local Behavior

Add tests in the `MagAOXApp` test area to cover:

- unresolved subscriptions retry according to the backoff schedule;
- receipt of a later `Def/Set` resets the retry state and resolves the property;
- a full refresh resets backoff and causes an immediate retry;
- once the 60 s clamp is reached, subsequent retries remain capped.

### Regression Checks

Verify that:

- existing subscription APIs remain unchanged;
- delayed publisher startup still converges without manual intervention;
- traffic from permanently unresolved properties becomes quiet;
- `logopt` / `hogopt` no longer drive high-rate discovery traffic.

## Risks

### Risk 1: Slower Detection of Newly Available Properties

Once backoff reaches 60 s, a newly available property may take up to about a minute to be rediscovered absent a refresh trigger.

Mitigation:

- treat 60 s as an acceptable recovery bound for this class of metadata;
- reset backoff on explicit refresh paths.

### Risk 2: Added State Complexity

The subscription manager becomes slightly more stateful.

Mitigation:

- keep the state limited to retry counters and timing;
- confine the logic to the existing set-subscription machinery in `MagAOXApp`;
- add focused tests.

### Risk 3: Server-Restart Detection Semantics

Backoff state must be reset after `indiserver` or a remote driver restart so retry does not remain artificially delayed.

Mitigation:

- reuse the existing full-refresh path to reset retry state;
- if needed later, add a slow periodic refresh safety net.

## Deliverables

The intended code changes are:

1. `MagAOXApp` support for exponential retry backoff on all unresolved set subscriptions.
2. Logging for long-unresolved properties without per-loop spam.
3. Tests covering retry schedule, clamp, and refresh reset behavior.
4. Verification against known high-traffic cases such as `logopt` / `hogopt`.

## Out of Scope for First Pass

- optional vs required subscription semantics;
- automatic classification of subscription criticality from config;
- changes to `indiserver` protocol behavior;
- broad redesign of INDI registration macros.

## Possible Later Enhancement

If exponential backoff for all subscriptions proves insufficient, a second-phase enhancement could add explicit optional/required semantics for monitored properties.

That should be treated as a follow-on feature, not part of the initial implementation.

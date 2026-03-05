# Incident Report: INDI Chained-Server `def*` Loop

- Date: 2026-03-05
- Branch under test: `dev-resurrector`
- Affected component: `apps/xindiserver` (`INDI/INDI/indiserver.c`)
- Status: Resolved on `dev-resurrector`; equivalent `dev` patch prepared, pending full system validation

## Summary

When two INDI servers were chained across hosts, property traffic for `pdu1.camwfs` entered a high-rate loop.  
`getINDI` (with no arguments) did not terminate after a single full property pass and repeatedly returned:

- `pdu1.camwfs.state=Off`
- `pdu.camwfs.target=Off`

## Impact

- Repeated property updates at high rate.
- `getINDI` output flood and non-terminating enumeration behavior.
- Unnecessary network and server load.

## Detection

Loop was observed in probe logs as repeated `defTextVector` for the same property bouncing between:

- remote driver path: `driver=pdu1@localhost:7627`
- chained client sockets on `:7624`

The same property (`dev=pdu1`, `name=camwfs`) appeared repeatedly in both RX paths.

## Root Cause

`def*` messages from remote drivers were being forwarded to chained-server clients and re-circulated through inter-server links.  
This created a reflection loop in the chained topology.

## Fix

Minimal fix was implemented in `q2Clients(...)` in `INDI/INDI/indiserver.c`:

- Add source context (`srcIsRemote`, `roottag`) to `q2Clients`.
- If message source is remote and tag is `def*`, skip forwarding to clients marked as chained servers (`cp->allprops == 2`).

This blocks the remote `def*` reflection edge while preserving normal client forwarding behavior.

## Files Changed

- `INDI/INDI/indiserver.c`
  - `q2Clients` signature updated
  - one forwarding guard added in `q2Clients`
  - callsites updated in:
    - `shutdownDvr(...)`
    - `readFromClient(...)`
    - `readFromDriver(...)`

## Validation

- Reproduced the original loop before fix in live tunneled environment.
- Applied patch and rebuilt `xindiserver`.
- Re-tested same scenario: loop stopped.
- User confirmation: issue resolved.

## Follow-Up (dev branch)

- After RTC improvement on `dev-resurrector`, similar but less severe loop behavior was observed on AOC/ICC where `dev` branch `indiserver` is used.
- A logically equivalent fix was ported to `dev` in the threaded `indiserver` implementation:
  - Branch: `jrmales/dev-indiserver-chained-def-loop`
  - Commit: `a784e57f`
  - Core behavior: suppress forwarding of remote-driver `def*` updates to chained clients.
- Full multi-host validation on AOC/ICC remains pending due to test configuration timing.

## Notes

- Debug probe instrumentation used during triage was removed.
- Final patch intentionally kept minimal to reduce regression risk.

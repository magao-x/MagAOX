# ICC indiserver High CPU Investigation (2026-03-11)

## Summary

On ICC, `indiserver` was observed to reach high CPU usage after some time. When this state occurs, plain `getINDI` appears to loop indefinitely.

Static analysis in this repository suggests the strongest cause is not a tight internal spin in `indiserver` itself, but persistent INDI discovery traffic caused by one or more MagAO-X apps repeatedly requesting properties they never successfully receive.

The clearest concrete candidate on ICC is `logopt` (`modalGainOpt`), which subscribes to loop properties that `loloop` does not appear to publish in this tree.

## Key Observation About `getINDI`

Plain `getINDI` is not a bounded one-shot snapshot here.

- With no arguments it defaults to `*.*.*`.
- Wildcard queries never satisfy `finished()`.
- The program exits only after a timeout with no new matching `def*` traffic.
- That timeout is reset whenever a new `def*` message arrives.

Relevant code:

- [INDI/INDI/getINDI.c](/home/jrmales/Source/MagAOX/INDI/INDI/getINDI.c#L230)
- [INDI/INDI/getINDI.c](/home/jrmales/Source/MagAOX/INDI/INDI/getINDI.c#L484)
- [INDI/INDI/getINDI.c](/home/jrmales/Source/MagAOX/INDI/INDI/getINDI.c#L508)
- [INDI/INDI/getINDI.c](/home/jrmales/Source/MagAOX/INDI/INDI/getINDI.c#L603)

This means `getINDI` "looping" is consistent with continuous `def*` activity from the server, not necessarily with a bug inside `getINDI`.

## Main Suspected Mechanism

`MagAOXApp` will keep sending `getProperties` for subscribed `set` properties until all expected properties have been seen at least once.

Relevant code:

- Main loop callout: [libMagAOX/app/MagAOXApp.hpp](/home/jrmales/Source/MagAOX/libMagAOX/app/MagAOXApp.hpp#L1985)
- Request generation: [libMagAOX/app/MagAOXApp.hpp](/home/jrmales/Source/MagAOX/libMagAOX/app/MagAOXApp.hpp#L3260)
- Completion on receipt of `Def/Set`: [libMagAOX/app/MagAOXApp.hpp](/home/jrmales/Source/MagAOX/libMagAOX/app/MagAOXApp.hpp#L3473)

If a subscribed property never arrives, the app will continue to issue `getProperties` indefinitely at its loop cadence.

That repeated request stream can drive repeated `def*` responses, which in turn keeps plain `getINDI` alive and can contribute substantial `indiserver` CPU load.

## Strongest ICC Candidate: `logopt`

ICC runs `logopt` according to [proclist_ICC.txt](/opt/MagAOX/config/proclist_ICC.txt).

`logopt` is `modalGainOpt`. In startup it subscribes to:

- `emgain`
- `psdTime`
- `psdAvgTime`
- `loop_state`
- `loop_gain`
- `loop_multcoeff`
- `loop_pcgain`
- `loop_pcmultcoeff`
- `loop_pcOn`

Relevant code:

- [apps/modalGainOpt/modalGainOpt.hpp](/home/jrmales/Source/MagAOX/apps/modalGainOpt/modalGainOpt.hpp#L941)

ICC `logopt` configuration points it at `loloop`:

- [logopt.conf](/opt/MagAOX/config/logopt.conf)

In this tree, the `loloop` implementation is `cacaoInterface`, and it publishes:

- `loop_state`
- `loop_gain`
- `loop_multcoeff`
- `loop_max_limit`

Relevant code:

- [apps/cacaoInterface/cacaoInterface.hpp](/home/jrmales/Source/MagAOX/apps/cacaoInterface/cacaoInterface.hpp#L317)

It does not appear to publish:

- `loop_pcgain`
- `loop_pcmultcoeff`
- `loop_pcOn`

This creates a likely permanent mismatch:

- `logopt` waits for `loloop.loop_pcgain`
- `logopt` waits for `loloop.loop_pcmultcoeff`
- `logopt` waits for `loloop.loop_pcOn`
- those properties may never be defined
- `logopt` keeps issuing `getProperties`

## Additional Review: RTC and AOC

### RTC: `hogopt` is the same pattern

RTC runs:

- `holoop` as `cacaoInterface`
- `hogopt` as `modalGainOpt`

Relevant config:

- [proclist_RTC.txt](/opt/MagAOX/config/proclist_RTC.txt#L3)
- [hogopt.conf](/opt/MagAOX/config/hogopt.conf#L1)

`hogopt` subscribes to the same predictive-control properties as ICC `logopt`:

- `loop_pcgain`
- `loop_pcmultcoeff`
- `loop_pcOn`

Relevant code:

- [apps/modalGainOpt/modalGainOpt.hpp](/home/jrmales/Source/MagAOX/apps/modalGainOpt/modalGainOpt.hpp#L941)

But `holoop` is also `cacaoInterface`, which in this tree only publishes:

- `loop_state`
- `loop_gain`
- `loop_multcoeff`
- `loop_max_limit`

Relevant code:

- [apps/cacaoInterface/cacaoInterface.hpp](/home/jrmales/Source/MagAOX/apps/cacaoInterface/cacaoInterface.hpp#L317)

So `hogopt` is a strong RTC analogue of the ICC `logopt` issue.

### RTC: latent naming mismatch if `hofilt` is enabled

`hofilt` is commented out in the RTC process list at present:

- [proclist_RTC.txt](/opt/MagAOX/config/proclist_RTC.txt#L7)

However, if it is enabled later, there is a likely naming mismatch:

- `modalFilter` publishes `loop_pcon`
- `modalGainOpt` subscribes to `loop_pcOn`

Relevant code:

- [apps/modalFilter/modalFilter.hpp](/home/jrmales/Source/MagAOX/apps/modalFilter/modalFilter.hpp#L586)
- [apps/modalGainOpt/modalGainOpt.hpp](/home/jrmales/Source/MagAOX/apps/modalGainOpt/modalGainOpt.hpp#L949)

That is a separate interface inconsistency even if the predictive-control feature is intended to exist.

### AOC: no equally clear fixed-code mismatch found

For AOC, this review did not find another comparably clear code/config mismatch among the fixed-function local apps in:

- [proclist_AOC.txt](/opt/MagAOX/config/proclist_AOC.txt#L3)
- [isAOC.conf](/opt/MagAOX/config/isAOC.conf#L1)

The AOC apps most likely to still exhibit this class of problem are the config-driven consumers such as:

- `labrules` (`stateRuleEngine`)
- `instgraph` (`xInstGraph`)

Those remain plausible because stale property keys in their config could produce the same retry behavior, but no obvious broken key was identified in this pass.

## Notes on `indiserver`

`indiserver` does contain logic to mitigate chained remote-server loops for generic `getProperties` and certain def forwarding paths.

Relevant code:

- Driver `getProperties` handling: [INDI/INDI/indiserver.c](/home/jrmales/Source/MagAOX/INDI/INDI/indiserver.c#L1084)
- Generic `getProperties` remote-device rewrite: [INDI/INDI/indiserver.c](/home/jrmales/Source/MagAOX/INDI/INDI/indiserver.c#L1404)
- Chained-client def suppression: [INDI/INDI/indiserver.c](/home/jrmales/Source/MagAOX/INDI/INDI/indiserver.c#L1592)

From static inspection, these paths make `indiserver.c` a less likely primary root cause than repeated traffic sourced by one or more applications.

## Notes on `xindidriver`

`xindidriver` contains a comment noting that its `select()` may not behave as intended because the FIFO is opened `O_RDWR`:

- [INDI/xindidriver/xindidriver.cpp](/home/jrmales/Source/MagAOX/INDI/xindidriver/xindidriver.cpp#L235)

However, the loop still proceeds into blocking `read()` calls, so this did not stand out as the best explanation for long-lived high CPU in `indiserver`.

## Most Likely Conclusion

The most likely explanation is that one or a few ICC applications are generating persistent `getProperties` traffic because they subscribe to properties that are absent, stale, or mismatched against current loop-driver interfaces.

`logopt` is the strongest specific suspect found in code/config review.

The same class of issue likely exists on RTC via `hogopt`, and there is also a latent RTC naming mismatch between `modalFilter` and `modalGainOpt` if `hofilt` is enabled.

## Recommended Runtime Checks

1. Temporarily run `indiserver` with higher verbosity and look for repeated `getProperties` from the same driver.
2. Confirm whether `logopt` is repeatedly requesting `loloop.loop_pcgain`, `loloop.loop_pcmultcoeff`, or `loloop.loop_pcOn`.
3. When high CPU occurs, capture a short `strace -f -p <indiserver-pid>` or `perf top`.
4. If the traffic hypothesis is correct, the hot path should be dominated by socket/FIFO `read()` and `write()`, not internal compute.
5. Verify whether ICC `logopt` should target a different loop device or whether its subscribed property names need updating.

## Possible Remediations

1. Fix the `logopt` to loop-device interface mismatch.
2. Add logging in `MagAOXApp::handleSetProperty()` for invalid or never-satisfied subscriptions.
3. Add rate limiting or backoff in `sendGetPropertySetList()` for properties that remain unresolved for long periods.
4. Audit other ICC apps using `REG_INDI_SETPROP` for stale property names.

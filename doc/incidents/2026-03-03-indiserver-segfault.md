# indiserver Segfault Investigation (2026-03-03)

## Summary

On Tuesday, March 3, 2026, `indiserver` crashed with repeated `SIGSEGV` events on `exao3` (and similarly observed elsewhere), followed by queue-backlog restart messages from `xindiserver` and then graceful `xindiserver` exit after detecting `indiserver` was gone.

Representative kernel message:

```text
[Tue Mar  3 13:27:25 2026] indiserver[4022920]: segfault at 7fa7f6ffd910 ip 00007fa9a3e8814d sp 00007fa94a7fb870 error 4 in libc.so.6[7fa9a3e28000+175000]
```

Representative `xindiserver` log symptom:

```text
Driver /opt/MagAOX/drivers/pupilCorAlign: 52428843 bytes behind in 659603 messages, restarting
```

## What Was Changed

File changed:

- `INDI/INDI/indiserver.c`

Branch created (per branch policy):

- `jrmales/indiserver-segfault-fix`

### Fix 1: varargs format-string UB in `openRemoteConnection()`

Several `Bye()` calls had format strings requiring `%s:%d:%s` but only passed `strerror(errno)`. This is undefined behavior and can crash in libc.

Patched calls now pass `host`, `port`, and `strerror(errno)` correctly.

### Fix 2: one-time fd close in backlog restart paths

When queue depth exceeded `maxqsiz`, code could repeatedly `close(dp->rfd)`/`close(cp->s)` from multiple paths. Under fd reuse, repeated close can hit unrelated descriptors.

Patched logic now:

1. grabs the respective queue lock,
2. swaps fd to `-1` once,
3. closes only the saved fd if valid.

This was applied in:

- driver queueing path (`q2Drivers`)
- snooping driver queueing path (`q2SnoopingDrivers`)
- client queueing path (`q2Clients`)

### Build Verification

`indiserver` rebuilt successfully with:

```bash
make -C INDI/INDI indiserver
```

## Deployment Status

Patched `indiserver` has been deployed to two machines (per operator note).

## Core Dump Findings So Far

On Rocky 9 target host, `coredumpctl` has an entry:

- Timestamp: `2026-03-03 13:28:05 EST`
- PID: `4022400`
- Signal: `SIGSEGV`
- Executable: `/opt/MagAOX/bin/indiserver`
- Core storage: `/var/lib/systemd/coredump/...zst`
- Status: **truncated** (8.2 MB), stack trace not actionable (`n/a` only)

## Next Steps (Planned)

1. Ensure future cores are not truncated:
   - Tune `/etc/systemd/coredump.conf` on Rocky 9:
     - `Storage=external`
     - `ProcessSizeMax` and `ExternalSizeMax` to large values (example: `8G`)
     - `MaxUse`/`KeepFree` to fit disk policy
2. Ensure launcher context sets unlimited core size:
   - `ulimit -c unlimited` in the same user/session context that starts `xindiserver` (not root unless root starts it).
   - If managed by systemd unit, set `LimitCORE=infinity`.
3. Verify tmux behavior:
   - start `tmux` **after** setting `ulimit`, or restart tmux server.
4. Wait for recurrence, then immediately collect:
   - `coredumpctl list indiserver`
   - `coredumpctl info <pid>`
   - `coredumpctl debug <pid>` (or export core + `gdb thread apply all bt full`)
5. If a full backtrace is still unavailable:
   - verify debug symbols for `/opt/MagAOX/bin/indiserver` (`-g`, not stripped, or split debuginfo installed).

## Quick Triage Commands (Rocky 9)

```bash
# Check whether core capture is configured and entries exist
cat /proc/sys/kernel/core_pattern
coredumpctl list indiserver

# Check active process core size limit
cat /proc/$(pidof indiserver)/limits | grep -i 'max core file size'

# Inspect specific crash
sudo coredumpctl info <pid>
sudo coredumpctl debug <pid>
```

## Notes for Future Follow-up

- The queue-backlog restart log line is likely a downstream symptom when `indiserver` is already destabilized.
- The patched UB + fd-close race were both high-risk for sporadic, hard-to-reproduce crashes with long intervals between events.
- If crashes persist with these fixes, next critical artifact is a full multi-thread backtrace from an untruncated core.

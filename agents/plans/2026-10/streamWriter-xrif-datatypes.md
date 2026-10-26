# streamWriter XRIF datatype compatibility

Prompt: The idle CPU fix worked, but some writers report XRIF encode error
`-100` when a producing process is restarted, possibly while saving or before
the first save. The example is `aol1_mmultfact` on 2026-10-26.

Reviewed: 2026-10-26, `jrmales/streamWriter-idle-cpu` at `a6d4d762`.

Status: Root cause reproduced locally; correction and permanent regressions are
implemented and validated. No instrument access, INDI traffic, dependency installation, or
configuration changes are involved.

## Finding

The installed XRIF API defines `-100` as `XRIF_ERROR_NOTIMPL`. With
`writer.compress=true`, streamWriter always configures image encoding as
PREVIOUS differencing, BYTEPACK reordering, and LZ4 compression. In this XRIF
version PREVIOUS only supports signed/unsigned 16-, 32-, and 64-bit integers,
and BYTEPACK only supports signed/unsigned 16-bit integers. Floating-point
images fail at differencing; wider integers fail at byte packing.

This is a pre-existing datatype compatibility defect, not a demonstrated
reconnect-state failure:

- The local producer in
  `/home/jrmales/Source/milk-xwcl/plugins/cacao-src/AOloopControl/modalfilter.c`,
  lines 1150–1156, creates `aol%lu_mmultfact` with
  `stream_connect_create_2Df32`, then writes through `array.F`.
- `/opt/MagAOX/config/aol1_mmultfact-sw.conf` arms writing at startup, uses a
  30-frame circular buffer and 10-frame write chunk, and leaves compression and
  the 10-second chunk timeout at their defaults. This is local configuration,
  not an inspection of the deployed process.
- The supplied log starts saving at `09:04:16.381659928` and reports its first
  encode failure at `09:04:26.392004651`, approximately 10.010 seconds later.
  That fits the partial-chunk timeout. It does not establish a restart as the
  trigger. The writer must have accepted data to reach this encoder call; an
  empty recording does not call XRIF encoding.
- The encoder selection is unchanged from before the idle-CPU fix. Earlier
  image round-trip tests and real-worker replacement regressions used UINT16,
  which supports both transforms. The replacement tests decoded timing records
  but did not validate float image payloads.

The local `/usr/local/include/xrif/xrif.h` is byte-identical to legacy headers
present at XRIF commits `cba92e1` and `5ddd15a`. The current XRIF checkout has a
newer incompatible `m_` member API, so it is not used to infer the installed
library's behavior. The checks below link `/usr/local/lib/libxrif.so` directly.
The deployed library revision has not been inspected.

## Reproduction

A temporary linked-library probe exercised the exact streamWriter method
selection with FLOAT32 data. Configuration, sizing, and buffer allocations all
succeed; encoding returns `-100`:

| Sequence | Encode result |
| --- | --- |
| New FLOAT32 handle, one frame | `-100` |
| Repeat FLOAT32 save, 15 frames | `-100` |
| Reallocate for the same FLOAT32 stream | `-100` |
| Reset and reconfigure FLOAT32 handle | `-100` |
| Reallocate and encode UINT16 | Success |
| Change the successful UINT16 handle to FLOAT32 | `-100` |

The methods remain unchanged after failure. Restarting or reinitializing the
encoder therefore cannot repair an unsupported datatype/method combination.
A separate 63-case probe varied seven datatypes, frame counts 1/2/512, and three
policies. NONE/NONE/LZ4 and fully uncompressed encoding both round-tripped bytes
exactly for UINT8, UINT16, INT16, UINT32, UINT64, FLOAT32, and FLOAT64.

Probe sources and logs are disposable artifacts at
`/tmp/streamWriter-xrif-notimpl-20261026/{probe,reconnect}.cpp`,
`results.log`, and `reconnect-results.log`. The observations here are the durable
record.

## Correction

Select the final preprocessing methods in `allocate_xrif()`, where the connected
source datatype is known, before allocating encoder buffers:

| Requested policy / source type | Difference | Reorder | Compress |
| --- | --- | --- | --- |
| Compression enabled, INT16 or UINT16 | PREVIOUS | BYTEPACK | LZ4 |
| Compression enabled, other supported types | NONE | NONE | LZ4 |
| Compression disabled | NONE | NONE | NONE |

This preserves the optimized 16-bit camera path and lossless compression for
other streams. It preserves floating-point representations, including NaN
payloads and signed zero, rather than doing floating-point arithmetic. It does
not change `m_compress`, so reconnection reselects appropriate methods even when
the stream datatype changes. The ordinary XRIF header stores the selected
methods; no archive-version or reader-format change is needed.

Do not retry unsupported preprocessing only after a failed encode: integer
differencing may already have modified the raw buffer before reordering fails.
Select supported methods before the first encode instead.

## Regression and validation record

Before the correction, permanent app tests reproduced the exact image encode
alert at line 2023 with code `-100`. The two new unit cases failed with 17 failed
assertions out of 270: compressed INT16/UINT16 and uncompressed variants passed,
while the other compressed scalar types failed. All four new real-worker
FLOAT32 scenarios also failed: an image archive was produced, but decoding its
image payload returned `-10001`. The previous timing-only decoder could not
detect that invalid image payload.

The corrected regressions cover:

- Forty scalar-type/compression/frame-count combinations (ten types, compressed
  or raw, and one-frame or full three-frame chunks), with byte-exact image and
  timing round trips. Float/double inputs include positive/negative zero, NaN
  payloads, infinities, and subnormal representations.
- Repeated allocation on the same handles through
  FLOAT32 → FLOAT32 → UINT16 → FLOAT32 → FLOAT64 → INT16 → UINT8 → UINT32 → FLOAT32,
  with compression enabled and disabled, full and partial writes, and decoded
  header method checks. This verifies both fallback selection and restoration
  of the 16-bit transforms.
- Both real workers saving a single FLOAT32 frame through a timed partial flush.
- Real FLOAT32 source deletion/recreation while idle, armed before the first
  frame, or active with one unsaved frame. Each scenario decodes image and timing
  payloads, checks every timing word, and compares image bytes exactly. Tests
  use shortened flush timeouts and bounded subprocesses; production defaults
  stay unchanged.

Fresh builds with `COVERAGE=1` passed:

| Executable | Test cases | Assertions |
| --- | ---: | ---: |
| `streamWriter_test` | 8 | 452 |
| `streamWriter_fault_test` | 18 | 1102 |
| `streamWriter_lifecycle_test` | 10 | 207 |
| `streamWriterSizing_test` | 1 | 34 |
| `streamWriter_entrypoint_test` | 1 | 11 |

Lifecycle counts are parent-process assertions; all child-process checks also
passed. The entrypoint target rebuilt the actual application before checking its
help/configuration interface. Build/invocation pattern, from `tests/`:

```sh
make -f Makefile.one t=../apps/streamWriter/tests/streamWriter_test.cpp -W ../apps/streamWriter/tests/streamWriter_test.cpp COVERAGE=1
GCOV_PREFIX=/tmp/streamWriter-xrif-final-gcov timeout 90s ../apps/streamWriter/tests/streamWriter_test
```

The app header and both modified test files were clang-formatted and checked
with `git diff --check`. Final logs are temporary artifacts:
`/tmp/streamWriter-xrif-notimpl-20261026/unit-{prefixed,final}.log`,
`/tmp/streamWriter-float-{before,after}-fix.log`,
`/tmp/streamWriter-lifecycle-float-full.log`, and
`/tmp/streamWriter-xrif-{fault,sizing,entrypoint}.log`.

## Limits and follow-up

- The evidence explains this error for the local FLOAT32 producer and default
  compression policy. If a deployed writer still reports it after this fix,
  collect its source datatype, `writer.compress` setting, actual linked XRIF
  revision, and stderr error details. Instrument-side collection remains a
  user-run task under `AGENTS.md` rule 0.
- Previous files associated with encode alerts may be invalid. The existing
  encoder error policy logs an alert and continues writing: a failed fresh
  FLOAT32 handle retains compressed size zero, and a failed type-changing
  reconnect can retain the previous successful compressed size (237 bytes in
  the probe). This correction prevents the unsupported method selection; it
  does not repair old archives or change general encode-error handling.
- Compression ratios for non-16-bit streams depend on LZ4 alone. Adding more
  preprocessing methods requires separate compatibility and performance work.

Related engineering record:
[streamWriter idle CPU investigation](streamWriter-idle-cpu.md).

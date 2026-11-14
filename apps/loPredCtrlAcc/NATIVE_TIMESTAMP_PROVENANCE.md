# Timestamp provenance for the first native 60-second recording

2026-11-14. Source review only, on personal workstation `ASTR-LT-428.local`.
No native application, stream, driver, INDI server or instrument was accessed.
No controller, producer, calibration, operational configuration or output code
changed. The recording and its process/build snapshots are not present locally.

The operator reports 480,018 ADC and 120,000 camera/modal records, all rejected
before clock conversion: ADC lacks same-host evidence; camera/modal clocks are
unknown. There are 55,427 retained REALTIME beacons, with median full bracket
width 0.05 us. These results explain why mapped camera-marker pairs are zero.
They do not yet test the numerical clock mapper, phase, or causal ADC coverage.

`clock_spec()` defaults missing `same_host` to false and takes missing clock
declarations from the summary (`tests/native_timing_audit.py:99-113`). Therefore
this exact rejection pattern is expected without an explicit provenance
manifest, or with an equivalent unresolved manifest. It is not evidence of
an observed clock mismatch or excessive beacon uncertainty. With every mapping
rejected, `four_to_one_phase` is unavailable. Any phase population in that same
report is the separately labelled `observer_receipt_phase`, unless it comes
from another run/manifest or external processing. The frontend polls ADC before
DM/camera/modal (`native_observer.cpp:452-503`), so receipt-phase alternation is
not automatically camera-publication jitter or physical ADC/WFS phase jitter.

## Evidence scope and inspected revisions

Source assignments identify what a particular build does. They do not identify
the writer, loaded library, relay path or clock namespace during a past run.
Public configuration identifies an intended deployment, not the configuration
actually active for that recording.

| Source | Inspected revision | Use and limit |
| --- | --- | --- |
| This DDSPC checkout | `3f9109d1469beb4124ae9ec11cec0480e829878a` | Native journal semantics and older camera/frameGrabber code; not an installed-camera build attestation. |
| Sibling accelSim worktree | `9d4c5a99518d00da2b2cbef4773d2b9eea708666` | Simulator timestamp assignments; runtime binding still needed. |
| Public MILK `dev` | `add4f667c501392447161f35be8b9c056ed40153` | Reference MVM/helper implementation, not the unprovided instrument `milk-xwcl` build. |
| Its pinned ImageStreamIO submodule | `fa4e66da61c596298d56d222fd5d761b462db0e6` | Reference metadata writer and clock macro. |
| Public CACAO `dev` | `796b606fb68f2a81518d88910ce60c96f50ace37` | Conventional reconstructor setup and stream names. |
| Public MagAO-X config `onsky` | `c673302edbf57a583d0661df240fe9afab8035f6` | RTC process list/reference configurations, not saved run configuration. |

Public source copies and per-file SHA256 ledger were saved only under
`/private/tmp/ddspc-provenance-sources-jmlqg8lo/`. The immutable links below
identify reviewed versions independently of that temporary directory.

## acc000: accelSim

In [accelSim.hpp:873-913](https://github.com/magao-x/MagAOX/blob/9d4c5a99518d00da2b2cbef4773d2b9eea708666/apps/accelSim/accelSim.hpp#L873-L913),
`acquireAndCheckValid()` waits on a monotonic timer or synchronized software
schedule, samples monotonic time, then calls
`clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp)` at line 897. It advances
the numerical vibration model afterward. Acquisition metadata denotes software
sample-generation start on the producer's host, not the nominal deadline,
physical ADC aperture or WFS exposure. `loadImageIntoStream()` copies the two
encoded UINT16 values at lines 949-957.

The frameGrabber path in this checkout samples `CLOCK_REALTIME` after copying
pixels, assigns `atime=m_currImageTimestamp`, increments `cnt0`, stores circular
plane timestamps/counters, clears `write` and posts semaphores:
`libMagAOX/app/dev/frameGrabber.hpp:1008-1041`. Both ADC fields are REALTIME
for this producer path. Publication is a software marker before publication
completion, not a hardware event. Circular `cnt1` is a plane index; `cnt0`
labels published frames. Neither is a WFS parent identity.

The matching sibling frameGrabber at `9d4c5a99:1105-1155` preserves these
semantics. It has an optional relay hook, but neither accelSim nor OCAM
implements that hook in the inspected revision; they take the explicit
REALTIME assignment at line 1123. This distinguishes them from a generic
timestamp-preserving relay using the same base class.

Both clock calls execute on the producer's host. Source assignments establish
no machine name. The test context and reference RTC configuration support exao2
as intended placement, but a recorded source-to-process/host binding is absent.

## camwfs: OCAM, EDT and frameGrabber

The reference [RTC process list:11](https://github.com/magao-x/config/blob/c673302edbf57a583d0661df240fe9afab8035f6/proclist_RTC.txt#L11)
selects `ocam2KCtrl` for `camwfs`; its
[configuration:7-13](https://github.com/magao-x/config/blob/c673302edbf57a583d0661df240fe9afab8035f6/camwfs.conf#L7-L13)
selects that shared-memory name. The local handbook's
`operating/daily_startup.rst:19-21` identifies RTC as exao2. These establish
intended placement, not writer evidence for the recorded interval.

In this checkout:

- `apps/ocam2KCtrl/ocam2KCtrl.hpp:1592-1594` passes
  `m_currImageTimestamp` into `edtCamera::pdvAcquire()`.
- `libMagAOX/app/dev/edtCamera.hpp:549-558` obtains two words from
  `pdv_wait_last_image_timed()`, rearms acquisition and copies them directly to
  seconds/nanoseconds. The wrapper does not sample a new acquisition clock.
- `ocam2KCtrl.hpp:1602-1645` checks the OCAM image number. frameGrabber's
  `cnt0++` is a separate publication sequence, not that raw camera number.
- frameGrabber writes the inherited SDK value to `atime` and samples its own
  REALTIME publication marker. `cnt1` is the circular plane index.
- `ocam2KCtrl.hpp:1716-1739` copies both timestamps and `cnt0` into
  `camwfs_sync`, then posts it. This creates neither an exposure marker nor
  a timestamp of the synchronizer's semaphore receipt.

The newer sibling camera path at `9d4c5a99` retains the SDK timestamp through
`ocam2KCtrl.hpp:2381` and `edtCamera.hpp:703-745`, with guarded result validation
and deferred rearm. Its sync copy is at `ocam2KCtrl.hpp:2534-2559`. Neither
version synthesizes exposure bounds; the loaded version still needs matching.

The [EDT acquisition contract](https://edt.com/api-2018/group__acquisition.html)
describes host system/gettimeofday time around DMA completion, including
interrupt/response latency. The
[API source:5190-5243](https://edt.com/api-2018/libpdv_8c-source.html)
retrieves the selected completed buffer's timestamp via `edt_get_timestamp`,
not the user thread's return time. Host REALTIME is a supported candidate for
the documented default mode; neither event qualifies exposure boundaries.

The two-word wrapper and older vendor prose require a matching-version check.
[Timestamp-mode definitions](https://edt.com/api-2018/libedt_8h-source.html)
include `EDT_TM_SEC_NSEC` and alternatives. The sibling engineering record
`agents/plans/2026-10/ocam2k-acquisition-publication-timing.md:790-845` documents
an inspected 5.6.8.0 SDK copy and seconds/nanoseconds decoder, but explicitly
does not attest the library/driver loaded during this recording. It is
corroboration, not loaded-version/mode evidence. Do not change units or assign
the active EDT domain from numerical resemblance alone.

## aol1_modevalWFS: producer selection matters

The conventional configured producer is **`wfs2cmodeval-1`**, invoking
`linalg.MVMmextrmodes` from `milklinalgebra`. The
[CACAO setup:12-38](https://github.com/cacao-org/cacao/blob/796b606fb68f2a81518d88910ce60c96f50ace37/AOloopControl/scripts/cacao-fpslistadd-MVMGPU_WFS2CMODEVAL#L12-L38)
selects input/trigger `aol1_imWFS2` and output `aol1_modevalWFS`.
The reference instrument configuration configures this process and comments out
`nnReconstructor` in its RTC process list. A historical operator trace in the
sibling `agents/plans/2026-10/dm-syncopation-debugging.md:364-375` also records
`camwfs -> aol1_imWFS2 -> aol1_modevalWFS`. Neither source establishes which
writer was active throughout this later 60-second observation.

For the reviewed conventional reference implementation:

1. [MVMextractModes.c:951-958,1044-1085](https://github.com/milk-org/milk/blob/add4f667c501392447161f35be8b9c056ed40153/plugins/milk-extra-src/linalgebra/MVMextractModes.c#L951-L1085)
   computes output and calls `processinfo_update_output_stream()`.
2. [The helper:13-76](https://github.com/milk-org/milk/blob/add4f667c501392447161f35be8b9c056ed40153/src/CommandLineInterface/processinfo/processinfo_update_output_stream.c#L13-L76)
   delegates to `ImageStreamIO_UpdateIm_atime(..., NULL)`. Its separate
   `CLOCK_MILK` trace timestamp is not the metadata timestamp writer.
3. [ImageStreamIO.c:2254-2289](https://github.com/milk-org/ImageStreamIO/blob/fa4e66da61c596298d56d222fd5d761b462db0e6/ImageStreamIO.c#L2254-L2289)
   samples `CLOCK_ISIO`, sets `writetime=ts` and `atime=ts` for NULL input,
   increments output `cnt0`, clears `write` and posts. Both fields describe
   output update, not inherited camera acquisition.
4. [ImageStreamIO.h:13-20](https://github.com/milk-org/ImageStreamIO/blob/fa4e66da61c596298d56d222fd5d761b462db0e6/ImageStreamIO.h#L13-L20)
   defaults `CLOCK_ISIO` to `CLOCK_TAI` when available, otherwise REALTIME,
   unless overridden. The compiled metadata-writer library determines this;
   a header on a different host or a `CLOCK_MILK` trace is not proof of it.
5. The helper copies trigger traces and stores upstream `cnt0` in
   `streamproctrace[0]`, not metadata `cnt1`. The reviewed MVM/helper does not
   propagate camera `cnt0` or acquisition into modal metadata. Its trigger
   counter is read at output update: without a frozen input-snapshot contract,
   even this trace is not automatically a certificate of pixels used by MVM.
   The current CSV does not retain those trace rows or writer PID.

There is an alternative writer. In this checkout,
`apps/nnReconstructor/nnReconstructor.hpp:518` attaches to an existing output;
`:701-705` sets `atime` to REALTIME at its input processing callback;
`:773-777` wakes a separate publishing worker. frameGrabber subsequently writes
REALTIME publication time and its own counters. The
[reference NN configuration](https://github.com/magao-x/config/blob/c673302edbf57a583d0661df240fe9afab8035f6/nnReconstructor.conf#L1-L13)
names the same modal stream. Its acquisition field means callback arrival,
not camera DMA completion. A stream name alone cannot identify its active writer.

The public MVM reference is not assumed identical to the instrument's customized
GPU graph/FP16 implementation. Its active `milk-xwcl` revision, helper and loaded
ImageStreamIO build must be matched before applying these reference semantics.

## Same-host provenance and the existing beacons

Reference RTC placement makes a local exao2 chain plausible. It is not inferred
from near-epoch values or rates. The journal saves backing/image identities,
counters and raw/observer timestamps, but not producer PID/executable/host,
stream process traces, relay history, dependency versions or clock namespace
information (`native_observer.cpp:62-70,550-608`). A mapped local file can be a
received/republished remote stream; its inode is not a producer-host certificate.
The ADC reader's receipt clock is not the producer clock.

Needed evidence is a run-associated writer/process and configuration record (or
an explicit operator attestation covering the interval), binding each stream's
identity to its active writer on exao2, with no timestamp-preserving remote
relay. The modal writer may differ from the stream's creator. Match executable
and loaded ImageStreamIO/EDT versions, compile-time clock selections and active
SDK timestamp mode. Later snapshots alone cannot prove an unchanged past run.

The 55,427 beacons are generated locally by `native_observer.cpp:42-59,467-469`:
MONOTONIC before, REALTIME now, MONOTONIC after. They are observer-host clock
relations, not accelerometer/WFS synchronization references. Their 0.05-us
median full bracket gives about 0.025-us midpoint half-width for a typical
beacon, before age/drift/neighbor-envelope terms. Median alone bounds neither
outliers nor clock steps or mapping coverage.

| Recorded field, if runtime provenance is confirmed | Existing REALTIME beacons |
| --- | --- |
| Local accelSim acquisition/publication | Usable subject to unchanged per-record mapping checks. |
| Local OCAM frameGrabber publication | Usable subject to those checks. |
| Local EDT default host-REALTIME acquisition | Usable after matching active SDK/driver/mode semantics. |
| Modal output using REALTIME, including the NN path | Usable after identifying that writer/build. |
| Modal output using TAI | Insufficient without an evidenced contemporaneous TAI/REALTIME relation. |
| Remote producer timestamps | Insufficient without a qualified cross-host clock relation. |

Do not relabel TAI as REALTIME or subtract an assumed 37 seconds. The Linux
host's TAI offset must be established for the recorded interval. No retained
TAI beacon has been reported. Schema 1's audit has no external TAI/REALTIME
transform input; new evidence for that relation would need a separately reviewed
analysis-only extension. Unknown native camera clocks did not erase raw
timestamps, but also did not record a new clock domain automatically. Existing
ADC REALTIME beacons can be shared by qualified same-host REALTIME sources;
a new acquisition is not required just for that.

## Manifest and next decision

`examples/native-timing-provenance-60s.partial.json` records source-backed ADC
REALTIME and camera-publication REALTIME candidates. All `verified` and
`same_host` fields remain false; false here means not established, not proof of
a remote producer. EDT acquisition and the active modal domains remain unknown.
It is a partial evidence ledger, not a qualified mapping manifest. Run/stream
hashes and identities remain unbound because actual artifacts are absent.
Additional documentation keys are ignored by the schema-1 audit; they grant no
permissions. No mapping, association or expected phase is certified.

After independent run evidence is provided, update a **new private copy** only
for proven fields. Each `evidence` string must name the run artifact or operator
attestation and matching build. Keep association and phase targets unverified;
do not infer them from matching counters. Rerun the audit with the unchanged
25-us uncertainty and 2-ms anchor-age limits:

```sh
export PYTHONDONTWRITEBYTECODE=1
DDSPC_TOOLS="$HOME/source/MagAOX-DDSPC/apps/loPredCtrlAcc/tests"
DDSPC_OBS="$HOME/ddspc-native-observe.9UlnxD/observations"
DDSPC_MANIFEST="/absolute/path/to/evidence-reviewed-private-provenance.json"
DDSPC_OUT="$(mktemp -d "$HOME/ddspc-provenance-audit.XXXXXX")"
python3 "$DDSPC_TOOLS/native_timing_audit.py" "$DDSPC_OBS" \
  --provenance "$DDSPC_MANIFEST" > "$DDSPC_OUT/timing.json"
```

The unchanged partial manifest should yield zero mapped samples and
`offline_prediction_timing_ready=false`. With local REALTIME ADC/camera fields
established, their publication-relative phase can already be diagnosed while
modal qualification remains blocked. Compare same-record camera markers only
when the EDT domain is established. Decompose phase increments as
`delta_ADC_acquisition - delta_camera_publication`; fixed pairing must retain
four ADC counter increments per camera frame. Inspect raw/mapped examples
before attributing the reported +/-105 us to publication, pairing or conversion.

Causal availability requires mapped acquisition/publication upper bounds and
actual frontend receipt by the selected deadline. The journal supplies
observer/publication-marker diagnostics, not qualified WFS processing deadlines
or exposure intervals. Modal atime described above does not repair that gap.
Replay training also needs independently confirmed applied inputs.

This simulator run does not qualify the real accelerometer's intended 4:1
phase lock. That needs actual synchronization reference IDs/times, delay
adjustments and acquisition event/clock evidence. Future corrections must use
the advertised dmtweeter-synco slot identity and input-ready cutoff, never
unscheduled DM writes. Approximately 110-us service in 125-us slots leaves
15-us nominal slack; neither these beacons nor this review qualifies that
schedule. All training and output permissions remain disabled.

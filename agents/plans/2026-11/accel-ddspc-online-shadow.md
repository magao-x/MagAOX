# Qualified online multirate shadow milestone

2026-11-14, personal workstation ASTR-LT-428.local, branch
`parkertjohnson1/accel_ddspc`. No instrument, network INDI, running application,
installed binary, production configuration, service or DM was accessed. No commits
or pushes are authorized. `apps/loPredCtrl` remains unchanged.

The requested online training path reuses native RLS/QR and the reference DDSPC
solve. Extend command width independently from measured WFS width: four distinct
125-us inputs per 500-us measured WFS frame, eight retained ADC feature channels,
rectangular future response, no averaging/interpolated WFS. Keep single-rate
standard defaults and an explicit matched four-slot WFS-only comparison setting.

Implementation: bounded nonblocking native observation fanout; platform-independent
qualified multirate assembler; separate ordinary-priority numerical worker; explicit
physical-evidence/exposure/source/schedule adapter seam; independent histories,
fit/rank/readiness/conditional scoring; fixed committed prefix in the existing solve;
precise compact INDI readback; disabled-by-default startup flag. No output interface.
Stride controls adaptation only, retains every input/history row, and changes the
per-second forgetting interpretation when gamma is below one. Soft budget overruns
latch a fault; this is not a hard WCET guarantee or an 8-kHz deadline-qualified path.

Source review cannot qualify current real inputs. EDT/frameGrabber markers do not
establish physical exposures; the actual modal producer and parent frame are not
attested; combined DM publication/SDK completion is not physical application. The
present native observer cannot construct the required qualified records. Do not add
config trust bits or train on shadow proposals. Current live telemetry must keep
training inhibited. Normal 2-kHz held inputs also cannot excite four independent
command-slot columns; empirical readiness must not be waived.

Validation and remaining blockers are described in
`apps/loPredCtrlAcc/ONLINE_SHADOW.md`. Tests use explicit synthetic provenance and
private transport/framework doubles, including the real application header, actual
learners and actual worker. Local numerical timings are throughput observations,
not telescope performance or synco deadline qualifications. Full suite, focused
sanitizers, native acquisition/INDI/safety regressions and confirmed-input replay
are required before review. Final results are recorded in that document after runs.

Completed locally: full regression 145 cases; native acquisition/INDI/safety 57;
confirmed-input replay 6; native tools Python 45; focused ASan/UBSan 19 cases with
no findings. Real Linux/GCC/ImageStreamIO link remains untested on this Mac.
Legacy's 15 reference files remain byte-identical. Local unpaced two-mode
stride-four QR fixture: 40.820-us average / 297.125-us maximum per synthetic WFS
record; no target deadline or 8-kHz scheduling qualification. Review also fixed
immediate candidate revocation on ADC loss/acquisition-time reversal, cleared old
mapped sensor retention on a new contract and preserved root provenance reasons
through initial raw-stream attachments. No change was installed or committed.

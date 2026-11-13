# Native observation only

2026-11-13. Start from committed replay checkpoint `1346b311` on
`parkertjohnson1/accel_ddspc`; personal workstation `ASTR-LT-428.local`.

Implement only disabled-by-default input observation, with no learner/predictor
input admission or hardware/Legacy actions. Retain ADC-only behavior when off.
Reuse the bounded ADC cursor API. Observe existing modal/camera/combined DM and
optional completion/SDK streams. Preserve raw clocks and source identities;
record losses and missing provenance rather than guessing exposure or application.

Reuse the portable `common/dmCompletion.hpp`, `dmTiming.hpp`, `timingStats.hpp`
from the accelSim worktree at `9d4c5a99`, with no driver, scheduler or publisher
integration. Decode software endpoint kinds separately. Metadata identity matches
are observation diagnostics, never physical application proof.

One native input worker owns existing-only mappings. A bounded queue separates
its fixed-size records from the local CSV writer; overflow and I/O failure are
explicit. Use an existing private output directory with exclusive new files.
Stop/join observation before destroying the borrowed ADC reader. Training and
all shadow record admission remain inhibited throughout this mode, including
INDI requests. The initial example is a finite 60-second observation.

Validate pure record accounting, missing/restarted streams, protocol errors,
clock mismatches, bounded ADC forwarding, writer overflow, application guards,
and unchanged offline replay. Build only on a local development host, using an
existing local Linux VM if available; do not run a native application or connect
to live streams. Record results in `NATIVE_SHADOW_TESTING.md` before a separate
feature commit. No deployment, install, push, services, or instrument connection.

Implemented and locally verified on 2026-11-13:

- Disabled-default native observer, finite input worker + exclusive private
  CSV/JSON writer with a bounded SPSC queue. Existing ADC cursor forwarding;
  read-only modal/camera/combined request and optional software endpoint snapshots.
- Strict private executable admission, inhibited record/training seams and
  read-only training property; native mode skips original WFS/FPS subscriptions.
- Exact raw IDs/timestamps, separate image/filesystem inode, unknown clocks,
  losses, restart/sequence diagnostics and explicitly unqualified associations.
- Private example, artifact verifier and operator-only build/transfer/start/
  diagnostics/60-second acceptance/rollback guide in `NATIVE_SHADOW_TESTING.md`.

Apple Clang offline results: full regression 123 cases;
focused guarded observation/application 30 cases / 10,438 assertions; ADC 15
cases / 448,098 assertions; confirmed-input replay 6 cases / 40,049 assertions.
Focused observation/application suite also passes ASan+UBSan and TSan, including
100,000 concurrent queue publications, overflow accounting and ADC/native
reconnection. Python orchestration 9 and config/artifact 2 cases pass. No
sanitizer findings. Clang-format applied with the repository style and minimum
line-comment spacing 0 to retain the mandated `//mutex scope` annotations.

Existing application replay benchmark at nominal 2-kHz WFS / 8-kHz ADC remains
passing: H15/A5 total-frame p50 102.916 us, p99 132.792 us, maximum 302.292 us;
0 / 1800 measured frames above 500 us. This is unpaced offline replay, not
native observation/journal or real-time scheduling qualification.

SHA256 comparison confirms ADC transport/calibration, DDSPC mathematical core,
original loPredCtrl, accelSim and historical tracked results remain unchanged;
accelSim worktree is clean. No native executable was launched, no instrument
stream/service was contacted, and nothing was installed/deployed/pushed.
Native Linux compile/link remains unverified: the personal Mac lacks native
ImageStreamIO/GCC14/Eigen3.4, and the build helper correctly refuses this host.
Fresh private native compilation and the operator's finite 60-second observation
remain required before considering a separate shadow-training implementation.
Latest-frame transport gaps and absent exposure/physical-application provenance
must be reviewed explicitly; vector summaries are not complete actuator history.


Git staging, commits and pushes are reserved for the user. No files were staged
and no commit was made. The native implementation is separate from the clean
committed replay checkpoint and ready for user review/staging.

Affected files (all application-relative paths below are under
`apps/loPredCtrlAcc/`):

- Application/build: `loPredCtrlAcc.hpp`, `Makefile`;
  `loPredCtrlAcc_native_observe.hpp/.cpp`.
- Observation transport/accounting: `native_observation.hpp/.cpp`,
  `native_observer.hpp/.cpp`.
- Portable shared protocol: `libMagAOX/common/dmCompletion.hpp`,
  `libMagAOX/common/dmTiming.hpp`, `libMagAOX/common/timingStats.hpp`.
- Tests/build fixtures: `tests/Makefile.offline`, `tests/Makefile.replay`,
  `tests/adc_transport_fixture.hpp/.cpp`,
  `tests/native_stubs/ImageStreamIO/ImageStreamIO.h`,
  `tests/native_linux_build.py`, `tests/native_linux_build_test.py`,
  `tests/ddspc_native_observation_test.cpp`,
  `tests/native_observation_app_test.cpp`,
  `tests/native_observation_tools_test.py`, `tests/verify_native_observation.py`.
- Handoff: `NATIVE_SHADOW_TESTING.md`,
  `examples/loPredCtrlAcc-native-observe.conf.example`, and this engineering plan.

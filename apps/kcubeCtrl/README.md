# K-Cube power and connection recovery

`kcubeCtrl` requires the external `tmcController` dependency with
`TMCC_IO_RECOVERY_VERSION >= 1`. The companion change is on
`jrmales/kcube-power-recovery`, based on upstream `6ba83aef` (functional commit
`8421603`; final companion revision `070043f`). Update that dependency
before building this app. The dependency remains external to MagAOX.

By default, the build uses `/opt/MagAOX/source/tmcController` and pkg-config's
libftdi1 flags. An alternate checkout can be selected with:

```sh
make -C apps/kcubeCtrl TMC_CONTROLLER_PATH=/path/to/tmcController
```

`FTDI_CFLAGS` and `FTDI_LIBS` can override header and linker flags when pkg-config
is unavailable. The dependency's tests directory is required for the app tests:

```sh
make -C tests -f Makefile.one t=../apps/kcubeCtrl/tests/kcubeCtrl_test.cpp \
  TMC_CONTROLLER_PATH=/path/to/tmcController
apps/kcubeCtrl/tests/kcubeCtrl_test '[kcubeCtrl]'
```

These tests use real app/protocol code with fake FTDI functions and a private INDI
fixture. They do not communicate with hardware.

With power management enabled, USB work requires both actual and target power On.
A power interruption or transport failure discards both connections and interrupted
commands. The app retries discovery and reinitializes with outputs disabled; a new
operator command is needed to enable or set them. Last measured voltage values are
retained with Idle/Alert status until fresh readings arrive.

| Configuration | Default | Behavior |
| --- | --- | --- |
| `connection.retryInterval` | 1000 ms | Delay between failed connection attempts; allowed 1..60000 ms |
| `timeouts.io` | 1000 ms | Individual USB operation timeout and overall reply deadline; allowed 1..60000 ms |
| `power.powerOnWait` | 0 s | Optional wait before a new power generation's first attempt; shared base limits apply |

Both axis serials must be present and distinct. A configured startup wait also
applies when the app starts with power already On. Unknown power blocks I/O;
disabling power management permits normal unmanaged operation.

The [engineering plan](../../agents/plans/2026-10/kcubeCtrl-power-recovery.md)
records the original findings, implementation choices, offline results, and
operator-only hardware follow-up. The reported memory crash still needs a core
backtrace for a confirmed diagnosis.

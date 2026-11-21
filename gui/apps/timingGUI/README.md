# timingGUI

Build with `make -C gui/apps/timingGUI` and run `gui/apps/timingGUI/bin/timingGUI`
from the repository root. The panel uses the standard workstation INDI connection
at `127.0.0.1:7624`.

The synco slider controls `dmtweeter-synco.enable`. The line below it shows the
phase from `health.reason` and active step count from `timing.active`.
**Reacquire** resets its
timing estimate and leaves synco disabled. **Zero Channels** clears the configured
destinations when the AO loop is open and synco is disabled. Both modulators have
editable amplitude and frequency, followed by a modulation slider and a
**Zero** request for use after stopping modulation. `tweeterSpeck` also has an
angle input in degrees. `tweeterMod` shows its current file name, with the full
path on hover. Enter a numeric target and press Return to send it.

| Display | INDI property and elements |
| --- | --- |
| camwfs acquisition | `camwfs.fg_timing.acq_fps` and `acq_jitter` |
| synco delivered | `dmtweeter-synco.timing.deliveredHz` while `enable.toggle` is On |
| dmtweeter write | `dmtweeter.dm_timing.send_fps` and `send_jitter` at SDK return |
| holoop-time WMA | `holoop-time.loop_timing.mean` and `stddev` when `valid=1` |

Rates are in FPS. Latency and RMS jitter are converted from seconds to microseconds.
Unavailable readbacks appear as `---`.

The socket-free regression tests run with an offscreen Qt platform.

```bash
cd gui/widgets/timing/tests
qmake6 -makefile timingPanel_test.pro
make -f makefile.timingPanel_test
QT_QPA_PLATFORM=offscreen ./timingPanel_test
```

# timingGUI

Build with `make -C gui/apps/timingGUI` and run `gui/apps/timingGUI/bin/timingGUI`
from the repository root. The panel uses the standard workstation INDI connection
at `127.0.0.1:7624`.

The synco slider controls `dmtweeter-synco.enable`. The line below it shows the
phase from `health.reason` and active step count from `timing.active`.
**Reacquire** resets its
timing estimate and leaves synco disabled. **Zero Channels** queues a clear of
the configured synco destinations with the AO loop open or closed. The button
is disabled while dmSyncopation is servicing the request. `tweeterSpeck` has
amplitude, separation (λ/D), and angle (degrees) inputs. `tweeterMod` has
amplitude and frequency inputs and shows its current file name, with the full
path on hover. Both modulators have a modulation slider and a **Zero** request
for use after stopping modulation. Enter a numeric target and press Return to
send it.

With loop lock on by default, `tweeterSpeck` advances one sparkle plane per
main AO frame at step 0. Synco holds the plane through the auxiliary steps.
The full pattern repeats every four AO frames. For faster playback, set
`tweeterSpeck.loop_lock.toggle` Off and adjust `tweeterSpeck.frequency.target`
in INDI.

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

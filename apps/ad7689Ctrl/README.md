# ad7689Ctrl

Standalone AD7689 acquisition for MagAO-X, following `mcp3208Ctrl`'s
`MagAOXApp`, `dev::frameGrabber`, and `dev::telemeter` architecture. Timer mode
and ImageStreamIO semaphore synchronization use the same ADC owner and ordered
scan implementation. No reference application or downstream consumer is changed.

## Build and offline checks

Build on a **non-instrument Linux development machine**, with the normal MagAO-X
dependencies, Linux spidev headers, pkg-config, and **libgpiod >= 2.0** development
headers/library. The C API is used; a Python GPIO package is insufficient.

```sh
make -C apps/ad7689Ctrl
make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/AD7689_test
make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/ad7689Ctrl_test
apps/ad7689Ctrl/tests/AD7689_test
apps/ad7689Ctrl/tests/ad7689Ctrl_test
# Run native publication separately in a fresh, private shared-memory namespace:
task_shm_dir=$(mktemp -d /tmp/ad7689-shm.XXXXXX)
MILK_SHM_DIR="$task_shm_dir" apps/ad7689Ctrl/tests/ad7689Ctrl_test "[ad7689Ctrl-native]"
rmdir "$task_shm_dir"
```

The real driver also has a portable fake-platform target, requiring only a C++20
compiler. It never opens GPIO or SPI hardware:

```sh
make -C apps/ad7689Ctrl/tests test
make -C apps/ad7689Ctrl/tests sanitize
```

The driver test substitutes the platform calls around the actual implementation,
with an independent ADC configuration/EOC/sample model. The app suite substitutes
the device interface and tests the real application controls, timing, and streaming
methods. Test-only GPIO/SPI headers apply only to the driver suite. Production
builds require the real Linux and libgpiod headers. Validation translation units
use `-fno-fast-math` to retain NaN/Inf checks.

## Confirmed initial profile

| Signal | Connection / setting |
| --- | --- |
| CNV | GPIO8, physical pin 24, persistent manual push-pull output, initially low |
| SCK | GPIO11, pin 23 |
| DIN | GPIO10/MOSI, pin 19 |
| SDO | GPIO9/MISO, pin 21 |
| SPI | `/dev/spidev0.0`, mode 0, MSB first, eight-bit words, `SPI_NO_CS` |
| GPIO endpoint | `/dev/gpiochip0`, chip-local offset 8; verify the target mapping |
| REF | External, pin 2, reported approximately 3.32 V |
| REFIN | Pin 3, reported approximately 3.32 V; verify the schematic/source |
| Supplies | Reported VDD/VIO approximately 3.3 V; common ground; COM grounded |

GPIO8 has been observed reserved by `spi0 CS0`. The user must resolve that kernel
ownership/pinmux before hardware commissioning. `SPI_NO_CS` suppresses automatic
chip select; it does not release a line reservation. A busy GPIO causes a clear
startup failure. The application never takes a line from another owner or falls
back to automatic chip select.

The example config uses a separate instance (`ad7689Ctrl`), separate output stream
(`accad7689_test`), IN0/IN1, 2000 frames/s, 1 MHz SPI, and a 10 us conversion hold.
No service, site configuration, active `acc000` producer, or production consumer
is installed or switched by this change. After independent hardware approval,
the user can select the example using the normal MagAO-X `-n` and `-c` options.
Agents must not run this application or hardware commands on an instrument.

## ADC protocol and timing

The implementation follows the [AD7682/AD7689 Rev. K datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ad7682_7689.pdf),
especially Tables 6 and 13 and Figures 39/41, and cross-checks the two historical
configurations against the [ADI RAC driver](https://github.com/analogdevicesinc/no-OS/blob/main/drivers/adc/ad7689/ad7689.c).
The scratch Python scripts are working hardware references, with their quarter
bandwidth setting deliberately replaced by full bandwidth for production.

Supported CFG fields are CFG=1, INCC=111 (unipolar relative to GND), selected
INx=0..7, BW=1 (full), REF=110 (external direct REF; internal reference, buffer,
and temperature sensor disabled), SEQ=00 (manual order), and RB=1 normally.
`CFG = 0x3c71 | (channel << 7)` is left-aligned into `CFG << 2` and transmitted
as two bytes. For example, IN0 sends `f1 c4`, IN1 `f3 c4`, and IN7 `ff c4`.
Receive exactly two bytes and decode `(rx[0] << 8) | rx[1]` as unsigned 16-bit.
There is no MCP3208 command byte or 12-bit mask.

Each useful conversion waits for the previous acquisition/cycle/quiet guards,
raises CNV, holds it high through conversion, lowers it, waits the SPI setup guard,
and transfers with CNV continuously low. At the reported 3.3 V operating point,
tCONV is at most 3.2 us, full-bandwidth tACQ is at least 1.8 us, and the cycle is
at least 5 us. Defaults add a 10 us high hold, 1 us low-to-SPI setup, and 1 us
post-SPI quiet interval. Guards use monotonic deadlines after successful GPIO/SPI
completion, so software delays extend intervals. No clocks occur while CNV is high.
A fallback relative high hold protects cleanup after a timing-clock failure.
Actual edge timing and the part/voltage/temperature grade need physical validation.

Call k reads the result of its own CNV edge. The CFG sent in that call takes effect
at a later EOC and controls conversion k+2. Startup conservatively discards twelve
conversions, including delayed 32-clock CFG readback and restoration of 16-clock
framing. Unknown-state cycles are at least 20 us. A two-entry active/pending CFG
history primes the first two ordered channels; while reading slot i, the driver
writes the configuration for slot `(i+2)%N`. Normal frames use exactly N useful
conversions for N enabled channels, with no recurring dummy conversion. Reopen or
reconfiguration repeats initialization. SPI/GPIO/readback failures invalidate the
history and entire staged frame. Three consecutive failed acquisition/recovery
attempts stop acquisition; legal constant rail codes alone are not errors.

## Configuration and stream contract

`config/example.conf` lists all device settings plus inherited timing settings.
`accel.numChannels=1..8` selects the prefix IN0..IN(N-1). Optional
`accel.channels=[7,0,4]` selects an ordered, unique list. An explicitly supplied
count must match that list; runtime count changes cannot change its length.
`accel.fps` is the frame rate and the per-enabled-channel sample rate. SPI/CNV
and the explicit mapping are startup-only settings; runtime rate/count requests
are validated and applied at a stopped frame boundary.

The frame is N-by-1 `UINT16`, in configured channel order, with full codes
0..65535. `adc.vref` is calibration metadata for the conversion helper; it does
not rescale the raw stream. ADC-pin voltage is `D * VREF / 65536`, so 32768
corresponds to 1.66 V at VREF=3.32 V. Sensor acceleration requires separate zero,
gain, and sensitivity calibration.

Both modes set ImageStreamIO `atime` to software scan-start immediately before
the first useful conversion sequence. This precedes the first GPIO-high request;
scheduling/guard/syscall delays separate it from the physical aperture. Channels
are sequential samples, not simultaneous. Framegrabber maintains publication
`writetime`, `cnt0/cnt1`, circular records, timestamp arrays and semaphores. No
individual-channel timestamps are added. `timingDiag.channel_readout_us` and
`adcHealth.scan_us` report scan service duration, not physical per-channel offsets.
Measure aperture skew and software-timestamp offset externally.

Existing INDI `fps`, `numChannels`, `alpha`, `synchroDelay`, source subscriptions,
and `timingDiag` names/units are retained. `channelMap.inputs` reports ordered
physical inputs. `adcHealth` adds conversions, warmup discards, hardware errors,
invalid frames, overruns, missed timer periods, recovery attempts, useful published conversions, and scan time.
Diagnostic snapshots cross the acquisition/control thread boundary under a small
mutex; GPIO/SPI and semaphore operations run outside it. Invalid controls leave
current settings intact. Only complete fresh frames advance publication counters.

The unchanged `accelFFCtrl` currently assumes two channels and rejects raw values
above 4095. Its range/calibration extension is a separate companion task. Do not
connect this producer to it before that task and consumer qualification. Preserve
a two-channel profile for that handoff and record mapping, VREF, timing, and scale.

## Rate admission and hardware qualification

A requested rate must fit the configured lower bound
`N * max(high + setup + 16/SPIclock + quiet, 5 us)`; synchronized mode also reserves
`cadenceGuard_ns` (default 20 us). Actual Linux ioctl, scheduling, scan, logging,
and publication overhead are additional. The admission test cannot prove throughput.

| Profile | Configured eight-input scan budget | 8 kHz per channel |
| --- | --- | --- |
| 1 MHz, 10 us high, 1 us setup/quiet | 224 us | Rejected |
| 2 MHz, 10 us high, 1 us setup/quiet | 160 us | Rejected |
| 4 MHz, 4 us high, 1 us setup/quiet | 80 us | Candidate requiring measurements |
| 8 MHz, 4 us high, 1 us setup/quiet | 64 us | Candidate requiring measurements |

A semaphore source limits scan cadence; a 2 kHz source cannot produce 8 kHz per
channel. Timer mode counts missed periods and does not replay expired deadlines
in a burst. Synchronized mode retains producer-ID gap tracking, measured period,
EMA delay control, phase offsets, cadence caps, stale-stream reopening, and timeout
handling. Measured impossible cadence skips the scan and records an overrun.

Hardware validation and deployment are user-run, separate work:

1. Record OS/kernel/libgpiod versions, GPIO mapping/owner/pinmux and SPI clock.
   Confirm schematic, external REF/REFIN wiring, voltage tolerances/grade, grounding,
   and input drive/settling. Resolve the known GPIO8 reservation before startup.
2. Capture CNV, SCK, DIN, and SDO during startup, scans, reconfiguration and stop.
   Check high/cycle/acquisition/quiet/setup intervals, no automatic-CS edges,
   CFG readback, ordered channels and exactly sixteen clocks per useful result.
   Measure software `atime` to CNV offset and each channel's aperture offset.
3. Apply independently measured safe DC levels at zero, quarter, half and near
   full scale; compare mean codes against `round(Vin/VREF*65536)` with clipping.
   Use distinct levels across inputs and alternate large steps to expose ordering
   and settling errors. Establish uncertainty-based tolerances before acceptance.
4. Start at the example profile; compare 2 MHz/10 us against the scripts' single
   input runs, explicitly recording the BW difference. Qualify 4/8 MHz and 4 us
   holds before claiming performance. For each 1/2/4/8-input profile claiming
   8 kHz per channel, collect at least 60 seconds idle and under representative
   load: mean 7920..8080 Hz, zero unexplained losses, and all explained losses,
   invalid frames, missed deadlines and discards counted. Report RMS/max/complete
   interval distribution, scan skew, GPIO/SPI/scan/publication time and CPU use.
   No unapproved numerical jitter/skew threshold is implied.
5. Test restart, transport failure/recovery, stream recreation, retained records,
   shutdown and semaphore pacing; confirm an independent reader sees full-width
   values and correct timestamp/counter behavior on `accad7689_test`. Consumer
   handover and production ownership remain separate approval steps.

If software-paced CNV cannot achieve an agreed profile, report the measured limit
and review hardware-paced/kernel/DMA alternatives separately. Eight-channel 8 kHz
operation is a target, not a software-only or datasheet-only qualification claim.

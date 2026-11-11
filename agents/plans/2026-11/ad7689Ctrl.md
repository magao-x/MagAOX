# Task Description

## Problem Statement
Create a new MagAO-X application named ad7689Ctrl using the existing mcp3208Ctrl application as the reference implementation.

The new application should preserve the existing functionality, control interfaces, data acquisition architecture, and MagAO-X integration of mcp3208Ctrl, but replace the MCP3208 12-bit analog-to-digital converter (ADC) readout with the Analog Devices AD7689BCPZRL7 16-bit ADC.

The application will run on a Raspberry Pi 5 connected to a custom ADC PCB and will be used to acquire accelerometer signals for real-time vibration sensing and control within MagAO-X.

## Discussion
The existing mcp3208Ctrl application provides the starting point for implementing ad7689Ctrl. The new application should reuse as much of the existing architecture as possible, including:

1. MagAO-X application framework and lifecycle
2. Configuration file structure
3. SPI communication infrastructure
4. Data acquisition and sampling loops
5. Shared-memory image stream (shmim) functionality
6. Telemetry, logging, and error handling
7. Existing control and monitoring interfaces
8. Application build and deployment structure

The primary modification is replacing the MCP3208 ADC-specific communication protocol with the AD7689 protocol.

### ADC Specifications

| Parameter | MCP3208 | AD7689 |
|-----------|---------|--------|
| Resolution | 12 bits | 16 bits |
| Channels | 8 | 8 |
| Interface | SPI | SPI-compatible with CNV control |
| Maximum digital code | 4095 | 65535 |
| Architecture | SAR | SAR |
| Configuration | SPI command | 14-bit CFG register |
| Conversion sequencing | MCP3208 protocol | AD7689 pipelined configuration/readout |

The AD7689 supports up to 250 kSPS under specified operating conditions. The initial application should target approximately 8 kHz per enabled channel.

Unlike the MCP3208, the AD7689 requires different SPI transaction framing, conversion timing, channel selection, and configuration handling.

### References

AD7689 Datasheet:
https://www.analog.com/media/en/technical-documentation/data-sheets/ad7682_7689.pdf

AD7689 Product Page:
https://www.analog.com/en/products/ad7689.html

Analog Devices Reference Driver:
https://wiki.analog.com/resources/tools-software/uc-drivers/ad7689

### Hardware Configuration

The application will initially operate on a Raspberry Pi 5 with the following hardware configuration:

- ADC: AD7689BCPZRL7
- ADC resolution: 16 bits
- ADC supply voltage: 3.3 V
- Digital interface voltage: 3.3 V
- External reference voltage: approximately 3.32 V
- SPI interface: Raspberry Pi SPI0
- Initial SPI clock: 1 MHz, configurable
- ADC channels: IN0-IN7
- Initial target sampling frequency: 8 kHz per enabled channel
- Analog input configuration: unipolar, referenced to GND
- Channel sequencer: initially disabled unless justified by performance requirements

The current hardware connects the AD7689 CNV signal to Raspberry Pi SPI CE0 (GPIO8, physical pin 24).

Codex must investigate whether this wiring supports valid AD7689 conversion and readout timing.

Unlike a conventional SPI chip-select signal, CNV initiates conversions and controls portions of the ADC communication timing. Therefore, the existing MCP3208 SPI transactions cannot simply be reused.

If the existing wiring cannot reliably support the required CNV timing, document the issue and propose the smallest necessary hardware or software modification.

### AD7689 Communication Requirements

Implement the AD7689 interface according to the datasheet, including:

1. Constructing and transmitting the 14-bit configuration register (`CFG[13:0]`).
2. Selecting individual analog input channels.
3. Configuring the ADC for unipolar measurements referenced to ground.
4. Selecting the correct external reference configuration.
5. Configuring the ADC bandwidth and channel sequencer.
6. Correctly controlling the CNV signal and SPI clock.
7. Reading the 16-bit conversion result from SDO.
8. Accounting for the delayed relationship between configuration writes and conversion results.
9. Performing initialization and dummy conversions after startup or configuration changes.
10. Preventing stale, invalid, or incorrectly associated samples from entering the output stream.

The AD7689 configuration and readout sequence must follow the datasheet rather than assuming that each SPI transfer immediately returns the conversion result for the channel requested in that transfer.

### Data Conversion

For the initial unipolar configuration, convert raw ADC codes to voltages using:

$$
V_{\mathrm{in}} = \frac{D}{2^{16}} V_{\mathrm{REF}}
$$

where:

- \(D\) is the unsigned 16-bit ADC output code, ranging from 0 to 65535.
- \(V_{\mathrm{REF}}\) is the configured ADC reference voltage.
- \(V_{\mathrm{in}}\) is the measured input voltage.

The raw 16-bit samples must be preserved without truncation to 12 bits.

Any conversion to voltage or physical accelerometer units should preserve the behavior and conventions of `mcp3208Ctrl`, updating only the ADC-specific scaling where necessary.

### Sampling and MagAO-X Integration

The application should preserve the acquisition and shared-memory streaming architecture of `mcp3208Ctrl`.

The initial performance goal is approximately 8 kHz per enabled channel, with configurable sampling rates and channel selection.

Codex should determine whether the existing application samples channels sequentially, produces a single multi-channel frame, or uses another acquisition structure, and preserve that structure unless AD7689 timing requires a change.

The resulting application must remain compatible with existing MagAO-X accelerometer processing and synchronization workflows.

### Existing Python AD7689 Test Scripts

Two Python scripts are available that were developed during initial testing of the AD7689 ADC on the Raspberry Pi 5:

- `plot_ts_IN0.py`
- `plot_psd_IN0.py`

These scripts demonstrate initial working implementations of:

- SPI communication using `spidev`
- Manual CNV control using `gpiod`
- AD7689 configuration register construction
- ADC initialization and configuration pipeline flushing
- 16-bit ADC readout
- High-speed acquisition targeting 8–10 kHz
- High-resolution timestamps and sampling statistics
- Conversion of ADC codes to voltages
- Time-series and power spectral density analysis

Codex should review both scripts before designing the C++ ADC interface.

Use these scripts as references for the hardware configuration and acquisition approach, but do not assume that every implementation detail is correct.

In particular, verify the following against the AD7689 datasheet:

1. SPI mode and transaction framing.
2. CNV pulse timing and GPIO control.
3. Configuration register bit definitions.
4. External voltage reference selection.
5. ADC configuration and conversion pipeline latency.
6. Channel selection and result association.
7. Conversion delays and acquisition timing.
8. Raw 16-bit data decoding and voltage scaling.

Note that the Python scripts use manual GPIO control of CNV with SPI chip-select disabled (`spi.no_cs = True`).

Determine whether this approach should be preserved in the C++ implementation or replaced with a more reliable method.

The new `ad7689Ctrl` application should preserve the MagAO-X architecture of `mcp3208Ctrl` while using the Python scripts as additional references for AD7689-specific communication.

Do not copy the Python acquisition loop directly if the existing MagAO-X application provides a more appropriate real-time acquisition architecture.

## Requirements

1. Create `ad7689Ctrl` as a separate MagAO-X application without modifying or replacing `mcp3208Ctrl`.

2. Preserve the existing application functionality, including configuration, startup/shutdown behavior, telemetry, logging, monitoring, and control interfaces wherever compatible.

3. Replace the MCP3208-specific SPI readout protocol with the correct AD7689 configuration, conversion, and data retrieval protocol.

4. Support full 16-bit unsigned ADC acquisition without truncation.

5. Support configurable acquisition of channels IN0-IN7, preserving the existing channel ordering and stream conventions.

6. Correctly implement CNV signaling, acquisition time, conversion time, configuration latency, and SPI timing according to the AD7689 datasheet.

7. Target approximately 8 kHz per enabled channel while allowing the sampling frequency to be adjusted through the application configuration.

8. Preserve compatibility with existing MagAO-X shared-memory image streams, updating the pixel datatype or metadata only where necessary.

9. Maintain existing frame timestamps, sampling counters, and timing diagnostics used by downstream vibration analysis and synchronization systems.

10. Handle SPI failures, initialization errors, invalid configurations, missed samples, and acquisition overruns.

11. Integrate the new application into the appropriate MagAO-X build, configuration, installation, and deployment infrastructure.

12. Document hardware connections, configuration parameters, ADC scaling, acquisition timing, and application operation.

13. Avoid modifying unrelated MagAO-X applications or shared infrastructure unless strictly necessary.

## Tests and Metrics

### ADC Readout Accuracy

Verify that 16-bit conversion results correctly represent known analog input voltages.

For unipolar operation:

$$
D_{\mathrm{expected}} \approx
\frac{V_{\mathrm{in}}}{V_{\mathrm{REF}}} 65536
$$

Clamp the expected result to the valid 16-bit code range.

For example, with \(V_{\mathrm{REF}} = 3.32\ \mathrm{V}\) and \(V_{\mathrm{in}} = 1.66\ \mathrm{V}\):

$$
D_{\mathrm{expected}} = 32768
$$

Test known voltages on multiple channels and verify stable conversion results.

### Sampling Performance

Measure:

- Mean sampling frequency per enabled channel
- Sampling period distribution
- RMS sampling jitter
- Maximum timing deviation
- Missed or delayed samples
- CPU utilization
- SPI transaction duration

Initial target:

$$
f_s \approx 8000\ \mathrm{Hz/channel}
$$

Corresponding sampling period:

$$
\Delta t = \frac{1}{f_s} = 125\ \mu\mathrm{s}
$$

Determine whether this rate can be sustained across all enabled channels.

Eight channels at 8 kHz each require 64,000 conversions per second, excluding additional configuration or dummy conversions.

### Channel Validation

Verify:

- Correct channel selection
- Correct channel-to-output mapping
- No off-by-one conversion association
- No stale samples after channel switching
- Correct handling of unused channels
- Correct initialization and configuration behavior

### Data Integrity

Verify:

- Complete 16-bit sample preservation
- Correct byte ordering
- Correct unsigned data interpretation
- Correct voltage scaling
- Correct shared-memory datatype and dimensions
- Compatibility with downstream MagAO-X analysis tools

### Integration Tests

Verify that `ad7689Ctrl`:

- Compiles successfully
- Starts and stops correctly
- Loads its configuration
- Initializes the SPI device
- Produces valid shared-memory data
- Reports failures appropriately
- Preserves expected logging and telemetry behavior

Where possible, implement hardware-independent unit tests for ADC configuration generation, SPI data decoding, and channel association.


## Scope and Caveats

- Use `mcp3208Ctrl` as the primary reference for application structure and MagAO-X integration.
- Do not modify the existing `mcp3208Ctrl` application.
- Do not redesign the acquisition architecture unless required by AD7689 hardware timing.
- Preserve existing interfaces and downstream compatibility wherever possible.
- Do not assume MCP3208 SPI transactions are compatible with the AD7689.
- Do not assume SPI CE0 automatically provides valid AD7689 CNV timing.
- Do not assume the ADC returns the conversion result for the channel selected in the current SPI transaction.
- Do not assume 8 kHz per channel is achievable across all eight channels without testing.
- Do not silently discard invalid conversions or incorrectly associate them with channels.
- Confirm the reference configuration and input scaling against the datasheet and actual PCB.
- Hardware-dependent validation must be performed on the Raspberry Pi 5 and AD7689 PCB.
- Any required hardware modifications must be documented and approved before implementation.

# Instructions to Agent

Analyze the above task and create a plan to implement a solution.

Document your findings below under "Agent Findings and Plan". Keep this document up to date as you work.

Review `AGENTS.md`.

Thoroughly inspect the existing `mcp3208Ctrl` application, including its source files, configuration, build integration, SPI communication, shared-memory streaming, and dependencies.

Identify which components can be copied unchanged and which require modification for the AD7689.

Read the AD7689 datasheet, especially the sections covering:

- Digital Interface
- Configuration Register
- General Timing
- Reading/Writing During Conversion
- Reading/Writing After Conversion
- Channel Sequencer
- Reference Selection

Pay particular attention to the AD7689 configuration pipeline, CNV control, SPI framing, and conversion timing.

Do not alter any text above "Agent Findings and Plan" below.

Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Planning review finalized 2026-11-11 after reviewing the user's answers to all five questions. **Status: ready for approval; no critical blocker to implementing the standalone producer and offline tests has been identified. Implementation has not started.** Hardware commissioning and production/consumer handover have the separate gates listed under Implementation Readiness. Only this section has been edited; the protected prompt and the user's five answers are preserved verbatim.

## Task Summary

Create a separate `ad7689Ctrl` application using the current `mcp3208Ctrl` lifecycle, framegrabber, telemeter, INDI properties, timer/semaphore acquisition modes, and raw-count stream contract. Replace the ADC device layer and its per-channel read assumption with an AD7689 scan operation that understands configuration latency and CNV timing. Leave `apps/mcp3208Ctrl/` unchanged.

The principal findings are:

1. The existing application samples IN0 through IN(N-1) sequentially and publishes one N-by-1 `UINT16` frame. It does not publish one frame per channel, perform simultaneous sampling, or convert counts to volts/acceleration. Its stream already accommodates all 16 AD7689 bits.
2. The user confirms **manual GPIO8 CNV control with SPI_NO_CS** as the initial implementation: GPIO CNV rises to sample, stays high through conversion, then falls for a mode-0 SPI read/configuration write. Retain GPIO8/pin 24 without rewiring. The reported `/dev/gpiochip0` offset 8 mapping and the release of its known `spi0 CS0` reservation must be checked before a hardware run. Use `mcp3208Ctrl`, not the Python recording loops, for the application architecture.
3. A script-style conversion call reads the conversion initiated by that call's rising edge, but its transmitted CFG takes effect later: CFG written after conversion `k` affects conversion `k+2`. This is configuration latency, not a SAR conversion pipeline. A two-conversion look-ahead channel schedule can produce N fresh useful samples with N conversions per frame after initialization; the earlier automatic-CE0 N+1/guard-conversion proposal is superseded.
4. At 3.3 V the applicable full-bandwidth ADC ceiling is 200 kSPS, with a maximum conversion time of 3.2 us, rather than the 5 V figures of 250 kSPS and 2.2 us. Use the low-voltage timing table throughout.
5. Eight channels at 8 kHz require 64000 useful conversions/s and at least 1.024 Mbit/s for the 16-bit words alone. A 1 MHz SPI clock cannot meet this goal. Even the scripts' 2 MHz clock plus their 10 us conversion-high wait requires at least 144 us for eight conversions before GPIO/SPI overhead, exceeding a 125 us frame period. Their single-channel runs do not establish eight-channel performance.
6. Preserve the `acc000` raw-frame format while testing on a distinct stream and INDI identity. `accelFFCtrl` currently rejects codes above 4095 and requires a two-element frame; range/calibration changes and any wider-frame adaptation belong to the **separate companion task requested by the user**, after independent producer validation. Do not alter that consumer or downshift AD7689 counts in this implementation.

### Inspection record and reuse decisions

| Inspected files | Findings and proposed treatment |
| --- | --- |
| `apps/mcp3208Ctrl/mcp3208Ctrl.hpp` | Header-based `MagAOXApp<true>` with `dev::frameGrabber` and `dev::telemeter`, using their interface macros. Copy its architecture and interfaces into a new app; change acquisition at the ADC boundary. Remove the unused second `adc` handle from the new copy. |
| `apps/mcp3208Ctrl/mcp3208Ctrl.cpp` | Main entrypoint only. Copy that structure with the new class/application name. |
| `apps/mcp3208Ctrl/dependencies/MCP3208.h`, `MCP3208.cpp` | RAII-style lgpio handle; defaults to SPI bus 0/CE0, mode 0, 1.35 MHz. `read()` sends three bytes and masks to 12 bits. Replace this dependency in the new app; do not reuse the command, result mask, or immediate channel association. |
| `apps/mcp3208Ctrl/Makefile` | Uses `OTHER_HEADERS`, `OTHER_OBJS`, and `Make/magAOXApp.mk`; links lgpio with hardcoded x86 library/rpath directories. Reuse the build structure, replacing the dependency and omitting those architecture-specific paths. |
| `apps/mcp3208Ctrl/tests/mcp3208Ctrl_test.cpp` | Catch2 tests substitute the device methods and cover configuration, callbacks, raw frame copies, timer/semaphore reads, delay modeling, timing EMAs, counters, and reconfiguration. Adapt those behavioral tests and add protocol-level tests of the real new device implementation. |
| `scratch/plot_ts_IN0.py`, `scratch/plot_psd_IN0.py` | Read both scripts completely without executing/importing them: they perform hardware I/O at module scope. Reuse their independent CNV/mode-0/16-bit framing approach after the datasheet checks below; do not copy their acquisition architecture, calibration denominator, unchecked GPIO mapping, or plotting/file-label assumptions. |
| `libMagAOX/app/dev/frameGrabber.hpp`, `telemeter.hpp`; `Make/magAOX.mk`, `magAOXApp.mk`, `common.mk` | Reuse existing publication, circular-buffer, timestamp/counter, telemetry, privilege, and installation machinery. Framegrabber publishes only on `acquireAndCheckValid()==0`; positive skips a frame, negative exits the acquisition loop. No shared base-class change is presently needed. |
| Root `Makefile`; `tests/Makefile`, `Makefile.one`, `tests.list`, `groups.dox` | Root ACC roles currently build `mcp3008Ctrl`, not `mcp3208Ctrl`. Tests register the MCP3208 suite through `tests.list`; there is no app-local test Makefile or MCP3208 example config in this checkout. Add explicit new app/test integration rather than assuming it already exists. |
| `apps/accelSim/config/example.conf`, `apps/accelFFCtrl/{README.md,accelFFCtrlModel.hpp,accelFFCtrlInput.hpp,config/example.conf}`; `agents/plans/2026-04/mcp3208Ctrl-synchro.md` | Reuse naming/calibration conventions and synchronization semantics where applicable. The simulator's continuous phase-reference mode differs from the MCP3208 app's one-scan-per-semaphore mode; do not copy that redesign. Downstream 12-bit admission and empirical calibration are concrete migration issues. |
| `setup/README.md`, `local/config.example.mk`, `doc/templates/app_template.md` | Setup automation lives in the separate `magao-x/magao-x-setup` repository. Provide an in-tree example and deployment instructions here; site service/config changes require the actual deployment repository/instance information. |

Additional gaps in the reference app must be addressed in the **new copy only**:

- `loadConfigImpl()` reads `accel.numChannels`, but `setupConfig()` does not register it. There is no registered initial acquisition-rate setting; `fps.device/property/element/tol` describe a subscription, not a numerical starting rate.
- Local and external channel-count callbacks change `m_numChannels` without resizing the vector/stream or requesting reconfiguration. Count and FPS callbacks accept invalid values, including zero FPS and fractional channel counts.
- `MCP3208::connect/read/disconnect` throw, while the application does not translate those failures into its documented return-code/recovery paths.
- Timer acquisition never assigns `m_currImageTimestamp`; synchronized acquisition assigns it just before the sweep. Framegrabber copies that field directly into ImageStreamIO `atime` and the retained-frame timestamp array. The timer loop can also return success after shutdown/reconfiguration without having acquired a new sample.
- Current timing diagnostics describe software trigger/read events, not measured hardware conversion edges. The synchronized delay controller budgets measured non-delay service time and a configurable guard; a slower ADC scan must be included in that budget.

### Python-script findings checked against the datasheet

The user reports these scripts were used in initial Pi 5 tests. Their code establishes the intended interface and settings; no captures, CSV measurements, installed library versions, or physical timing traces accompanied them. The following distinguishes the actual executable settings from comments and from verified protocol requirements.

| Script detail | Finding and implementation consequence |
| --- | --- |
| SPI/GPIO setup in both | `SpiDev.open(0,0)`, mode 0, 8-bit words, 2 MHz, `no_cs=True`; `/dev/gpiochip0`, line offset 8, active-high output initially low. CNV is separately controlled through gpiod. Adopt Linux spidev with `SPI_NO_CS` and the libgpiod v2 C API; make chip/line configurable and verify the actual Pi mapping/ownership. `SPI_NO_CS` alone does not guarantee the SPI driver releases its GPIO CS reservation. |
| `convert()` in both | CNV high -> record monotonic time -> busy-wait at least 10000 ns -> CNV low -> two-byte full-duplex transfer -> decode `(rx[0]<<8)\|rx[1]`. This is a valid no-busy RAC structure when the conversion, acquisition, first-bit setup, and quiet-window requirements are met. Keep the result associated with the channel already acquired before this call's rising edge, not the channel in this call's TX word. |
| CFG fields in both | `CFG=1`, `INCC=111`, `REF=110`, `SEQ=00`, `RB=1` match manual unipolar GND-referenced/external-direct-reference/no-readback operation. `CFG<<2` and MSB-first byte construction are correct. The user now confirms external REF on pin 2 and the intended `REF=110` selection; its exact source, REFIN connection, and voltage tolerances remain schematic/commissioning checks. |
| Bandwidth in both | Both actually set `BW=0` (quarter bandwidth). At 3.3 V the aggregate ceiling is 50 kSPS, so this setting can support an 8-10 kHz single-channel test but cannot support eight channels at 8 kHz. Use `BW=1` (full bandwidth) in the new production configuration and enforce its timing; do not carry over quarter-bandwidth timing assumptions. |
| Actual channel/CFG | The time-series script sets `CHANNEL=1`, giving quarter-bandwidth `CFG=0x3cb1`, TX `f2 c4`, despite its IN0 header/CFG comments. The PSD script sets `CHANNEL=0`, giving `CFG=0x3c31`, TX `f0 c4`. Full-bandwidth equivalents are IN0 `0x3c71`/`f1 c4` and IN1 `0x3cf1`/`f3 c4`. Use evaluated fields, not filenames or comments, as the oracle. |
| Initialization in both | One configured `convert()` followed by 20 zero-TX calls, then a 50 ms pause. Sending zeros has CFG[13]=0 and retains configuration; it does not request channel 0. Twenty flush conversions conservatively hide startup latency for one fixed channel but neither verify CFG/readback nor test channel switching. Replace the magic count with explicit warmup validity/history and readback; use conservative timing while the power-up BW/CFG state is unknown. |
| Voltage scaling in both | They divide by 65535. The required transfer-function scaling is `D*VREF/65536`: maximum code is one LSB below VREF. Preserve unsigned raw counts and fix scaling in the new documentation/helpers/qualification analysis; do not reproduce the scripts' gain error or change these scratch files during planning. |
| Timestamp/deadline loop | `perf_counter_ns()` supplies a useful monotonic software timestamp before the GPIO-high request, not the physical CNV edge. Late absolute deadlines cause immediate back-to-back calls with no explicit missed-deadline count. GPIO/Python overhead may happen to supply settling time, but that is not a timing guarantee. Enforce ADC timing during warmup and catch-up/recovery too; keep the existing MagAO-X timer/semaphore structure and explicitly report overruns. |
| Run identity | The time-series script actually requests 8000 Hz for 5 s, 40000 samples, on IN1; headers/CSV/PNG names mention IN0, IN4, 2 kHz, or 0.1 s. The PSD script requests 10000 Hz for 30 s on IN0, while labels/filenames claim 8 kHz/20 s. Do not infer measured performance or channel identity from those strings. |
| PSD/export | The PSD script prints CSV filenames but contains no CSV-saving call; only its PNG is saved. It passes the mean software sample rate to Welch without checking uniform sampling, so deadline catch-up/jitter can invalidate a nominal uniform-rate spectrum. Future qualification must save real timestamps/raw data/run settings, report gaps and the interval distribution, and validate timing before interpreting PSDs. No plotting or SciPy dependency belongs in the production ADC app. |
| Failure/cleanup | The scripts rely on exceptions/finally; GPIO/SPI setup happens before the try block, so a setup failure can miss that cleanup. The new device owns both resource types, checks every operation, safely unwinds partial initialization, and invalidates a complete scan on any uncertain conversion/transfer. |

The recommendation is to preserve manual CNV because it follows the reported working interface, gives explicit control of conversion timing and sample onset, and avoids depending on automatic-CE0 high gaps. It does **not** guarantee Linux userspace jitter or all-channel throughput. Kernel/hardware-paced acquisition remains a conditional follow-up if measured requirements cannot be met.

### External references checked

- [AD7682/AD7689 datasheet, Rev. K](https://www.analog.com/media/en/technical-documentation/data-sheets/ad7682_7689.pdf): Tables 2/4/6/13 and printed pages 3, 6-9, 20, 22-28, 30-34; especially Figures 39 and 41-43. These provide the electrical/timing limits, CFG encoding, initialization, RAC/RDC/RSC distinctions, and sequencing behavior used below.
- [ADI product page](https://www.analog.com/en/products/ad7689.html) and [reference-driver documentation](https://wiki.analog.com/resources/tools-software/uc-drivers/ad7689); [ADI no-OS implementation](https://github.com/analogdevicesinc/no-OS/blob/main/drivers/adc/ad7689/ad7689.c) and [header](https://github.com/analogdevicesinc/no-OS/blob/main/drivers/adc/ad7689/ad7689.h). The driver explicitly tracks two historical configurations for RAC and left-aligns CFG in a 16-clock transfer. Use this as a cross-check, not as a source of Raspberry Pi timing guarantees or a new no-OS dependency.
- [lgpio SPI implementation](https://github.com/joan2937/lg/blob/master/lgSPI.c): its transfer wrapper submits one `SPI_IOC_MESSAGE(1)`, 8-bit words, with zero transfer delay; it exposes no composite-message timing control. It also reports the requested byte count for any nonnegative ioctl result.
- [Linux spidev documentation](https://docs.kernel.org/spi/spidev.html), [Raspberry Pi SPI documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#serial-peripheral-interface-spi), and reference Raspberry Pi Linux `rpi-6.12.y` [SPI core](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/drivers/spi/spi.c), [spidev implementation](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/drivers/spi/spidev.c), [UAPI](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/include/uapi/linux/spi/spidev.h), and [DesignWare controller](https://github.com/raspberrypi/linux/blob/rpi-6.12.y/drivers/spi/spi-dw-core.c). Pi 5 uses RP1/DesignWare rather than the older BCM2835 controller. The target's installed kernel, device tree, and actual waveforms remain unverified.
- [libgpiod v2 line-request API](https://libgpiod.readthedocs.io/en/v2.3/core_line_request.html), [line settings](https://libgpiod.readthedocs.io/en/v2.3/core_line_settings.html), and [Linux GPIO character-device API](https://docs.kernel.org/userspace-api/gpio/chardev.html): use one persistent output request with explicit physical polarity and checked set-value/release operations, not shell GPIO commands or legacy sysfs. GPIO line numbers are offsets within the selected chip.

## Key Assumptions

- All five questions have been answered. The software decisions below are settled for this plan; implementation still requires the user's explicit approval. No agent will execute on or control instrument hardware; the user performs Pi/PCB measurements and deployment steps under AGENTS.md rule 0.
- Confirmed board configuration: AD7689 external reference approximately 3.32 V on REF/pin 2, REFIN/pin 3 also measured at approximately 3.32 V, VDD and VIO approximately 3.3 V, and COM grounded. Use **REF=110**, with the internal reference, buffer, and temperature sensor disabled. The REFIN measurement is not a reason to enable the internal buffer. Verify the exact reference source/REF-to-REFIN routing, input filtering, and rail/reference tolerances before hardware qualification; these do not prevent implementing the agreed mode.
- Confirmed interface: SPI0 `/dev/spidev0.0`, mode 0, 8-bit words, manual active-high CNV on GPIO8/pin 24, normally low between completed reads, and `SPI_NO_CS`. Use libgpiod >=2.0 through its C API and configurable chip/offset, with the reported `/dev/gpiochip0` offset 8 as the initial profile. The user enabled `dtparam=spi=on`; that alone does not establish that the known GPIO8 CS reservation is released. No alternate automatic-CS backend or rewiring is part of the initial implementation.
- Preserve one sequential multichannel scan per timer event or synchronization semaphore event. In synchronized mode the sample rate follows the producer's semaphore cadence: an 8 kHz setting does not multiply a 2 kHz producer into 8 kHz acquisition.
- Retain the default two-channel count and 2000 FPS starting behavior for compatibility/commissioning; expose an explicit configurable initial rate and document the 8000 FPS target. Impossible rate/channel/SPI combinations are rejected instead of silently reducing the rate or changing the SPI clock.
- Default enabled channels remain the contiguous prefix IN0..IN(N-1). Add an optional ordered channel list for noncontiguous acquisition; output element `i` contains that list's physical input. Preserve ordering and expose/document the active mapping rather than sorting it implicitly.
- Full bandwidth, sequencer disabled, ground-referenced unipolar operation, confirmed external-direct REF selection, and no configuration readback are the normal acquisition settings. Startup verification temporarily enables readback, then restores normal framing and primes the history.
- The confirmed performance goal is 8000 frames/s, hence 8000 samples/s for every enabled channel. Initial qualification requires mean rate within 1% (7920..8080 Hz at this target) and zero unexplained losses over 60 s, with all losses/overruns counted. Report RMS jitter, maximum timing deviation, the interval distribution, and inter-channel skew; no numerical jitter/skew limit has been imposed. The 1 MHz bring-up setting, 2 MHz/10 us script profile, 4/8 MHz qualification settings, and candidate 4 us high hold remain distinct explicit configurations.
- The confirmed frame contract is one valid ordered multichannel `UINT16` frame per trigger, with software scan-start `atime` in timer and semaphore modes, scan-duration diagnostics, and measured/characterized timestamp-to-CNV offset. Individual-channel or hardware-aperture timestamp interfaces are outside the initial scope.
- The initial delivery is an independently validated Raspberry Pi 5 producer, using the existing MagAO-X build/instance conventions with the distinct `ad7689Ctrl` identity. The example uses `framegrabber.shmimName=accad7689_test`; this is a concrete nonconflicting test-name choice, not a production handover. Keep default channels IN0/IN1 and document their mapping. Consumer integration, actual site/service files, and replacement of an `acc000` owner are deferred work with their own prerequisites.
- Raw unsigned counts remain the production stream. Reference voltage is explicit calibration metadata/configuration, not a multiplier applied to that stream. At `VREF=3.32 V`, one ideal ADC count is approximately `50.65918 uV`; upstream signal-conditioning gain and accelerometer sensitivity remain separate calibration factors.
- Application implementations follow AGENTS.md rule 21: declarations plus out-of-class inline definitions in the new `.hpp`, with only main in `.cpp`. The standalone ADC dependency follows the existing `.h`/`.cpp` separation. These local conventions resolve the apparent tension with the general preference for out-of-header implementations.

## Requirements

1. Preserve `MagAOXApp<true>`, `FRAMEGRABBER_*`/`TELEMETER_*`, logging, `telem_fgtimings`, thread priority/cpuset configuration, timer/semaphore selection, and `fps`, `numChannels`, `alpha`, `synchroDelay`, and existing `timingDiag` element names/units.
2. Preserve N-by-1 `_DATATYPE_UINT16` frames, native `uint16_t` sample copies, circular-buffer behavior, ImageStreamIO publication counters/semaphores, and synchronization-producer counter tracking. Do not truncate, right-shift, or reinterpret raw samples as signed.
3. Implement AD7689 CFG generation, MSB-first transport, explicit conversion/acquisition guards, initialization, configuration/result association, and complete-frame validation. A transport or association failure invalidates the entire pending frame.
4. Make SPI bus/device index and clock, CNV chip/line and safe timing guards, startup FPS, channel count/list, and reference voltage configurable; validate configuration and local/external runtime requests through the same checks. SPI is fixed to mode 0/MSB-first/CS disabled, and unsupported analog modes stay out of the initial public interface.
5. Initialize valid software acquisition timestamps in both acquisition modes. Preserve existing read-start timestamp semantics and identify their offset/uncertainty relative to CNV; do not label modeled edges as measured hardware timestamps.
6. Report GPIO/SPI faults, intentional startup/recovery discards, invalid frames, rate overruns, and recovery state separately from producer-ID gaps and successful publication counters. The normal manual-CNV scan has no per-frame guard conversion. No stale or partial data may become a successful frame.
7. Integrate the standalone producer build, tests, documentation, and example configuration without replacing MCP3208/MCP3008 deployment. Use a distinct application identity/test stream, retain the `acc000` format, and defer `accelFFCtrl` range/calibration changes and production site/service activation to their separate work.
8. Qualify the configured 8 kHz-per-channel target with mean rate within 1% and zero unexplained sample losses over 60 s; report RMS jitter, maximum timing deviation, interval distribution, and scan skew without inventing additional pass thresholds. An analytical budget or single-channel Python run does not qualify all enabled channels.

## Questions and Points of Clarification

All five questions below have been answered and reviewed. Their original wording and the user's responses are retained as the decision record; they are not pending requests. The accepted choices are incorporated throughout this plan. Only the implementation approval remains pending for the in-repository producer scope; remaining commissioning/deployment checks are listed under Implementation Readiness.

1. **Reference and PCB:** Does approximately 3.32 V feed REF directly, or REFIN through the internal buffer? Please identify the reference source, actual ADC VDD/VIO rails, input buffering/filtering, and PCB connections around REF/REFIN/COM. At 3.3 V, buffered REFIN is limited to `VDD-0.5 V`, so 3.32 V there is unsuitable. For direct REF, Table 4 lists a maximum of `VDD+0.3 V`, while the pin description recommends `0.5 V..VDD`; resolve the nominal 3.32/3.3 V relationship and tolerances before final electrical acceptance rather than declaring the board invalid or assuming absolute-maximum limits guarantee accuracy. Recommended software assumption is direct REF with reference/buffer/temperature circuitry disabled (`REF=110`).

ANSWER: The AD7689 uses an external reference of approximately 3.32 V connected directly to REF (pin 2), with REFIN (pin 3) also measured at approximately 3.32 V. The ADC VDD and VIO rails are both approximately 3.3 V, and COM is grounded (0 V). The intended software configuration is REF=110, using the external reference directly with the internal reference, buffer, and temperature sensor disabled. The exact reference source, REF/REFIN PCB connections, and input filtering should be verified against the schematic. Since REF is slightly above the nominal VDD, the voltage tolerances should also be checked to confirm compliance with the datasheet.

2. **Pi software and GPIO ownership:** Which Pi OS/kernel, libgpiod/Python-spidev versions, and SPI/device-tree setup were used for the scripts? Confirm that `/dev/gpiochip0` offset 8 maps to physical GPIO8 on the intended system and how SPI0 CS reservation was disabled/reassigned. The recommended implementation preserves manual GPIO CNV plus `SPI_NO_CS`; no rewiring is proposed. Please provide any available run output/captures, with new logic-analyzer timing captured by the user during commissioning. Missing captures limit qualification, not the ability to design offline tests.

ANSWER: The Python scripts ran on a Raspberry Pi 5 using `spidev` and `libgpiod` for SPI communication and manual CNV control.

- **SPI:** SPI0 (`/dev/spidev0.0`), mode 0, 2 MHz, 8 bits per word.
- **CNV:** GPIO8 (physical pin 24), controlled manually through `/dev/gpiochip0` using `gpiod`.
- **Chip select:** `spi.no_cs = True`, disabling automatic SPI chip-select control.
- **Device tree:** `dtparam=spi=on` was enabled in `/boot/firmware/config.txt`.
- **GPIO ownership:** Earlier tests showed GPIO8 reserved by `spi0 CS0`. The scripts require GPIO8 to be released from SPI ownership before manual control.
- **Software versions:** Exact Pi OS, kernel, `libgpiod`, and `spidev` versions have not been recorded.

The recommended implementation is to preserve **manual GPIO CNV control with `SPI_NO_CS`**, without rewiring. GPIO8 mapping and ownership should be verified on the target Pi.

The Python scripts provide acquisition and timing diagnostics, but no logic-analyzer captures are currently available. These can be collected during hardware commissioning.

3. **Performance:** Review the conservative 10 us CNV-high starting guard and the separate script-reproduction profile at 2 MHz. The original 1 MHz bring-up setting remains available; qualify 4/8 MHz or a reduced high guard (initial candidate 4 us, never below the datasheet bound) before using a higher admitted rate. Are there required numerical jitter, deadline-miss, and scan-skew limits? Recommended provisional qualification is mean rate within 1% and zero unexplained losses over 60 s, with the full interval distribution reported. Single-channel 8/10 kHz Python targets are not proof of all-channel performance.

The initial AD7689 Python tests targeted **8–10 kHz single-channel acquisition**, using a 2 MHz SPI clock and a conservative 10 µs CNV-high conversion guard.

- **SPI speed:** 2 MHz used in Python testing; 1 MHz retained for initial bring-up.
- **CNV timing:** Initially 10 µs; a 4 µs guard can be evaluated after verifying datasheet timing requirements.
- **Higher SPI speeds:** 4 MHz and 8 MHz can be tested to improve acquisition performance.
- **Target sampling rate:** 8 kHz per enabled channel (125 µs sampling period).
- **Jitter:** No strict numerical limit established yet; minimize RMS jitter and report the full sampling interval distribution.
- **Deadline misses:** Target zero unexplained missed samples.
- **Scan skew:** Minimize and measure inter-channel acquisition delays; no numerical limit established yet.

For initial qualification, require **mean sampling rate within 1% of the target and zero unexplained sample losses over 60 seconds**, while reporting RMS jitter, maximum timing deviation, and sampling interval distributions.

The existing Python tests demonstrate a single-channel acquisition approach but do not establish sustained 8 kHz performance across all eight channels.

4. **Channels and timestamps:** Confirm the optional explicit ordered channel-list interface and the one-frame-per-trigger behavior. Recommended frame `atime` remains a software scan-start time (now populated in timer mode too), with measured scan duration and separately qualified CNV offset. If downstream processing requires hardware-aperture or individual-channel timestamps, approve that interface change separately.

The `ad7689Ctrl` application should preserve the existing `mcp3208Ctrl` acquisition and shared-memory streaming behavior.

- **Channel selection:** Support an optional explicit ordered channel list (e.g., `[0, 1, 2, 3]`), with the default matching `mcp3208Ctrl`.
- **Frame behavior:** Produce one multi-channel frame per acquisition trigger, containing one valid sample from each enabled channel.
- **Frame timestamp (`atime`):** Use the software scan-start timestamp, including in timer mode.
- **Scan duration:** Measure and report the time required to acquire all enabled channels.
- **CNV offset:** Measure and characterize the delay between the software timestamp and actual CNV triggering.
- **Channel timestamps:** Individual-channel or hardware-aperture timestamps are not required initially.

The goal is to preserve downstream MagAO-X compatibility while accurately characterizing acquisition timing. Any changes to the timestamp interface should be proposed separately for approval.

5. **Downstream compatibility and deployment:** Is `accelFFCtrl` a required initial consumer, and should its configurable ADC range/calibration extension be included in the implementation approval or handled as a companion task? Which Pi build role, INDI instance, output stream name, and site configuration/service repository should be targeted? Recommended producer defaults remain independent of any existing `acc000` owner; automatic deployment must not overwrite an active producer.

The `ad7689Ctrl` application should preserve compatibility with existing MagAO-X accelerometer acquisition and processing infrastructure.

- **Downstream consumer:** `accelFFCtrl` should remain compatible, but any required ADC range or calibration modifications should be handled as a separate companion task.
- **Build target:** Raspberry Pi 5, following the existing `mcp3208Ctrl` build configuration and deployment conventions.
- **INDI instance:** Follow the existing `mcp3208Ctrl` configuration, using a distinct `ad7689Ctrl` device identity.
- **Output stream:** Maintain compatibility with the existing `acc000` stream format, but use a separate stream name during initial testing to avoid conflicts.
- **Site configuration:** Follow the existing MagAO-X configuration and service repository conventions. Confirm the specific deployment files before implementation.
- **Deployment:** Do not automatically replace or overwrite an active `acc000` producer.

The initial priority is to implement and validate `ad7689Ctrl` independently on the Raspberry Pi 5. Integration with `accelFFCtrl` and production deployment can be completed after successful hardware and acquisition testing.

## Tests

### Hardware-independent tests to implement after approval

Use Catch2 and the existing central test build. Stub app-level device methods for interface tests; separately exercise the **real** ADC `.cpp` through test-local transport/clock wrappers. The latter must model CNV edges, CFG application at EOC, acquisition, and sample capture independently, not simply return a value keyed by the latest transmitted channel.

| Test group | Specific cases and required result |
| --- | --- |
| CFG/serialization | Check all channels 0..7 against fixed expected full-bandwidth words, including IN0 `0x3c71`/`f1 c4`, IN1 `0x3cf1`/`f3 c4`, and IN7 `0x3ff1`/`ff c4`. Cross-check the scripts' quarter-bandwidth IN0 `0x3c31` and IN1 `0x3cb1` calculations as inspection fixtures, without requiring an unsupported runtime mode. Verify field positions, `CFG<<2`, padding, RB polarity, invalid channels, and a zero-TX CFG[13]=0 hold that retains a previously selected nonzero input. |
| Full-width decoding/scaling | Decode `0000`, `0001`, `0fff`, `1000`, `7fff`, `8000`, `abcd`, `ffff`; preserve every bit and byte order through `loadImageIntoStream()`. Check unsigned midscale and `D*VREF/65536`, including 1.66 V/3.32 V = 32768 and maximum code mapping below VREF. Use the existing raw interface; test a conversion helper only if one is actually added. |
| Pipeline/scan association | Give each input a distinct, conversion-varying code. Each call raises CNV and reads that call's conversion; CFG sent in call k controls call k+2. Verify two-slot look-ahead, 1-/2-/8-channel scans, noncontiguous/reordered lists, repeated frame boundaries, and idle pauses: N useful apertures/N calls per normal frame, all after that frame's trigger. Poison earlier conversions to expose stale-frame reuse. |
| Startup/reconfiguration | Begin with undefined device state/readback; poison startup words. Verify at least the required dummy conversions, conservative warmup discards, configuration verification, and transition back to normal 16-clock reads before any publication. Repeat after reopen, scan-list/count change, transport failure, and acquisition restart. No old generation/sample may escape. |
| Timing/GPIO/SPI ordering | Use a fake monotonic clock and independent GPIO/SPI event log. Assert mode 0/MSB-first/8-bit words plus `SPI_NO_CS`, one persistent active-high GPIO request initially low, one controlled high/low pair per useful conversion, CNV high through EOC, first-bit setup, acquisition/edge-spacing/quiet minima, and exactly 16 clocks with CNV low. No automatic-CS event may trigger conversion. Test 10 us and qualified 4 us holds, a >=20 us startup cycle while BW is unknown, too-short/invalid timing rejection, and safe behavior under simulated scheduler delays. Only physical measurements establish actual edge timing. |
| CFG readback | Model delayed readback enable/disable, 32 clocks with 16 data bits followed by 14 CFG bits and padding, and matching against the historical configuration. A wrong channel/reference/BW/SEQ/RB response must prevent startup/force recovery. |
| Fault handling/resources | Inject GPIO-chip open/line-request failures, EBUSY/permissions/wrong-offset errors, allocation or SPI-mode setup failure after a line is claimed, GPIO high/low write failures, negative/short SPI results, EINTR with uncertain progress, malformed readback, clock failure, disconnect, and cleanup failure. Verify rollback of each acquired resource, best-effort stable CNV shutdown without an extra rising edge, whole-frame rejection, pipeline invalidation, bounded retry, and a nonthrowing destructor. No blind replay of a possibly completed conversion. |
| Config/runtime controls | Verify default/override/registered keys, GPIO path/offset and timing settings, and consistent startup rate calculations. Reject FPS <=0, NaN/Inf, unsafe timing, out-of-range SPI clocks/reference values, counts outside 1..8 or nonintegral, empty/duplicate/out-of-range lists, and count/list conflicts. Check local/external callbacks, pending reconfiguration, coherent vector/stream dimensions, and preservation of valid current settings after rejection. |
| Confirmed hardware profile | Verify that the example/loaded settings specify SPI0 device 0, mode 0/MSB-first/8-bit words plus `SPI_NO_CS`, `/dev/gpiochip0` line 8, `adc.vref=3.32`, and REF=110 with internal circuitry disabled. Exercise the known GPIO8-busy case as a clean startup failure; enabling SPI or setting `SPI_NO_CS` must not be modeled as automatically releasing that line. Check separate 1 MHz bring-up and 2 MHz/10 us comparison profiles. |
| Timer/semaphore integration | Port the existing behavior tests for signed delay wrapping, EMA/alpha, cadence caps/anti-windup, producer period/ID gaps, stale/recreated streams, timeouts, and mode transitions. Check nonzero timer `atime`, no frame on shutdown/reconfiguration/timeout, and one complete new scan per wake. An externally supplied 2 kHz semaphore must yield at most 2 kHz scans. |
| Publication/accounting | Verify N-by-1 `UINT16`, exact raw values, `atime` and retained timestamp arrays, framegrabber `cnt0/cnt1`, and local/sync-written counters advancing only on publication. Inject mid-scan GPIO/SPI failure and overrun; verify explicit invalid/missed/warmup counters without a partial frame or fabricated replacement sample. Late deadlines must not cause an uncounted catch-up burst; useful CNV count equals N per normal published frame. |
| Timestamp and identity contract | Check the distinct `ad7689Ctrl` instance and `accad7689_test` example stream, with no implicit selection/replacement of `acc000`. With a fake clock/event log, assert `atime` is software scan-start before the first useful CNV-high request in both modes, independent of SPI/read completion; verify scan-duration diagnostics and ordered channel mapping without introducing individual-channel timestamp fields. |
| Local integration | On a Linux development workstation/VM with MagAO-X dependencies, run a fake-GPIO/SPI producer with a local ImageStreamIO reader and synthetic semaphore source; verify startup/stop, stream recreation, retained records, logging, and telemeter hooks without opening real devices or using INDI hardware tunnels. Run sanitizers on new resource/fault/configuration paths where supported. |

Add `ad7689Ctrl_unit_test` under `application_unit_test` in `tests/groups.dox`; use `libXWCTest::ad7689CtrlTest`, per-TEST_CASE Doxygen briefs, and `tests/testXWC.hpp`. Hide harness-only symbols with `\cond`; use local `#ifdef AD7689CTRL_TEST_DOXYGEN_REF` references to the real API under test, enclosed by clang-format guards. Do the equivalent for the ADC suite. Account for the repository's `-ffast-math` default: compile validation code/tests with `-fno-fast-math` so nonfinite rejection is meaningful.

### User-run hardware qualification

No agent runs these steps on the Pi or instrument. After approved implementation, provide the user a repeatable capture procedure and collect the returned measurements:

1. **Electrical/timing:** Use the confirmed REF/pin 2 ~3.32 V, REFIN/pin 3 ~3.32 V, VDD/VIO ~3.3 V, COM=GND, and REF=110 profile. Before a hardware run, check the schematic/source/filtering, REF/REFIN connection and voltage tolerances, reported GPIO-chip/line identity, and release of the known `spi0 CS0` reservation. Record OS/kernel/libgpiod/spidev versions and pinmux/device-tree configuration. Then capture startup, single/multichannel operation, restart, and selected rates; measure high time/periods, first-clock setup, quiet/acquisition intervals, byte framing, readback, and timestamp-to-CNV offset. Verify `SPI_NO_CS` introduces no unintended GPIO8 edge. Captures are a commissioning deliverable, not an implementation prerequisite.
2. **Accuracy/channel identity:** With safe, independently measured DC inputs near zero, quarter-scale, half-scale, and near full-scale, compare mean code with `clamp(round(Vin/VREF*65536),0,65535)`. Use distinct levels on channels 0/1/7 and all enabled inputs to expose swaps; alternate widely separated levels to test mux settling. Report offset, gain, noise, and instrument/reference uncertainty. Do not demand exact single-code agreement from an analog converter; set acceptance tolerances from the combined error budget before the test.
3. **Performance:** Begin with the agreed 1 MHz low-rate bring-up, then reproduce the scripts' actual single-input targets (IN1/8 kHz and IN0/10 kHz, 2 MHz, 10 us high). Label the production BW=1 difference, using a separate approved harness if exact quarter-bandwidth comparison is required. Qualify 1/2/4/8-channel scans at 8000 FPS with explicitly configured guards/SPI speed; evaluate 4/8 MHz and a 4 us high hold only after timing checks. For each setting claimed to meet the goal, require mean per-enabled-channel rate in 7920..8080 Hz and zero unexplained losses over at least 60 s, idle and under representative load. Count/report all missed deadlines, explained losses, invalid frames, and warmup discards; report RMS jitter, maximum deviation, full interval distribution, scan skew, GPIO/SPI/scan/publish time, and CPU use. No separate numerical jitter/skew threshold is specified. Save raw/timestamps/settings and use physical-edge timing for jitter/CNV offset; mark a failing channel-count/profile unqualified rather than silently reducing its rate.
4. **Recovery:** User tests safe board power/reconnect and application restart/reconfiguration. Verify re-priming, stream generation/metadata changes, no stale samples, and bounded recovery. Valid constant zero/full-scale data must not alone be classified as disconnect.
5. **Stream/consumer boundaries:** For this producer delivery, verify an independent local reader sees the `acc000`-compatible geometry/datatype/metadata contract on `accad7689_test`, including unsigned codes above 4095. Keep a two-channel profile available for the current two-axis processing contract. The separate `accelFFCtrl` companion must update range/calibration admission and qualify its selected input geometry/mapping before actual consumer handover. Production stream ownership, retained records, and synchronized `atime` interpretation must be checked at that later stage; this plan does not claim the unchanged consumer accepts full-range data today.

## Implementation Plan

### 1. Proposed source/documentation change list

All entries are future changes after approval. No application files are changed during this planning pass. This change list covers the standalone producer, its offline verification, and in-tree example/documentation only; actual site/service deployment and `accelFFCtrl` modifications are deferred by the user's fifth answer.

| Path | Proposed work |
| --- | --- |
| `apps/ad7689Ctrl/ad7689Ctrl.hpp` (new) | Adapt the reference class, configuration/lifecycle/INDI handling, both acquisition modes, diagnostics, and raw streaming. Replace per-channel synchronous reads with one validated ADC scan, add rate/channel validation and safe reconfiguration, and fix the inherited timer timestamp/false-success paths. |
| `apps/ad7689Ctrl/ad7689Ctrl.cpp` (new) | Main entrypoint only. |
| `apps/ad7689Ctrl/dependencies/AD7689.h` (new) | Device/configuration/result declarations, named CFG masks, CNV timing/chip/line settings, channel schedule and two-slot CFG history, portable serialization helpers, and error contract. Keep libgpiod/Linux implementation details out of the public interface where practical. |
| `apps/ad7689Ctrl/dependencies/AD7689.cpp` (new) | Direct Linux spidev open/configure/transfer/close with `SPI_NO_CS`, libgpiod v2 persistent CNV request and high/low writes, all timing guards, readback/priming, N-call scan association, and partial-resource rollback. Both acquisition modes use this one ADC owner. |
| `apps/ad7689Ctrl/Makefile` (new) | Build the ADC object with `OTHER_OBJS/OTHER_HEADERS` and `Make/magAOXApp.mk`; discover libgpiod >=2.0 flags via pkg-config, link its C library, and retain system spidev headers. No no-OS/Python dependency or MCP3208 x86 rpath. Apply finite-validation compiler requirements locally. |
| `apps/ad7689Ctrl/tests/ad7689Ctrl_test.cpp`, `apps/ad7689Ctrl/tests/AD7689_test.cpp` (new) | Adapt app regressions and add real-driver protocol/transport fault tests. The driver suite can include the implementation through test-local syscall wrappers; app tests substitute device methods as the reference suite does. |
| `apps/ad7689Ctrl/tests/stubs/gpiod.h` (new) | Minimal test-only declarations for the used v2 API; the ADC suite supplies fake implementations/state, so offline tests cannot request GPIO. Limit the stub include path to that suite and also compile the production object against real v2 headers. |
| `apps/ad7689Ctrl/config/example.conf`, `apps/ad7689Ctrl/README.md` (new) | Encode the confirmed REF=110/VREF=3.32, GPIO8/manual-CNV/SPI_NO_CS settings, distinct `ad7689Ctrl` identity and `accad7689_test` stream, two-channel/low-rate defaults, and labeled performance profiles. Document 8 kHz acceptance, software timestamps, GPIO ownership/electrical checks, future consumer handoff, and user-run commissioning. |
| Root `Makefile` | Register the new Pi-targeted app alongside existing ACC applications without replacing `mcp3008Ctrl`; retain standalone `make -C apps/ad7689Ctrl` as the explicit build entrypoint. Include generic Linux `all_buildable_apps` coverage when spidev headers and libgpiod >=2.0 are available. No host-specific role/instance/service file is guessed or activated. |
| `tests/tests.list`, `tests/Makefile.one` | Register the two suites; add only necessary dependencies/test compile flags and any narrowly scoped driver-test linkage. |
| `tests/groups.dox` | Register the new application test grouping. |
| This plan | Record approved decisions, implementation progress, actual test commands/results, and remaining hardware qualification; never edit its protected prompt. |

**Deferred companion task, excluded from this producer implementation:** `apps/accelFFCtrl/accelFFCtrlModel.hpp`, `accelFFCtrlInput.hpp`, `accelFFCtrl.hpp`, `config/example.conf`, `README.md`, and corresponding `tests/accelFFCtrlModel_test.cpp`, `accelFFCtrlInput_test.cpp`, `accelFFCtrlGap_test.cpp`, and `accelFFCtrl_test.cpp` are likely future touchpoints. That task must add a validated configurable ADC maximum/range while retaining the existing 4095 default, update ordinary/single-gap admission, establish measured zero/gain/sensitivity, and test full-range codes. The current reader requires width=2 and height=1; initially route a two-channel profile, with any wider-frame selection/adaptation separately designed in the companion. Record the producer's ordered physical inputs, software `atime` semantics, measured VREF, `D*VREF/65536` scale, retained-stream metadata, and qualified rate/skew for that handoff. No `accelFFCtrl`, simulator, or unrelated-app source changes are part of this plan's implementation approval.

### 2. ADC configuration and word framing

The normal 14-bit CFG, interpreted using datasheet Table 13, will be:

| Bits | Setting | Purpose |
| --- | --- | --- |
| 13, CFG | 1 | Explicitly update the register for manual channel selection. |
| 12:10, INCC | `111` | Unipolar input referred to ADC GND; not COM-referenced `110`. |
| 9:7, INx | physical input 0..7 | Select the channel named by the scan schedule. |
| 6, BW | 1 | Full bandwidth, needed for the proposed aggregate throughput. |
| 5:3, REF | `110`, confirmed software selection | External REF/pin 2 directly; internal reference/buffer/temperature sensor disabled, including with the reported REFIN voltage. |
| 2:1, SEQ | `00` | Disable hardware sequencing; retain explicit arbitrary channel ordering. |
| 0, RB | 1 normally; 0 for verification | Readback is active-low. |

Thus `CFG(channel) = 0x3c71 | (channel << 7)` for normal operation. Send `CFG<<2` as two explicitly ordered bytes, MSB first, mode 0 (`CPOL=CPHA=0`), with eight bits per byte and CNV continuously low across both bytes. Receive `D=(uint16_t(rx[0])<<8)|rx[1]`; no MCP3208 start byte, nibble mask, sign extension, or result shift remains.

Use named fields/masks and checked inputs rather than silently masking invalid channels. Do not use the all-ones factory preload: it enables different reference/sequencer settings. CFG readback during startup/commissioning uses four bytes/32 clocks to hold the conversion's 16 bits plus its associated 14-bit CFG and two trailing padding bits. Track the historical RB state while enabling/disabling readback, verify relevant configuration fields, then re-prime the normal 16-clock schedule.

The hardware sequencer is unnecessary to reach the ADC's underlying capacity and cannot represent an arbitrary channel list; enabling it would not remove CNV/conversion timing requirements or the 1 MHz bandwidth limit. Defer it unless measurements establish a specific benefit and the user approves the changed sequencing policy.

### 3. Manual CNV/SPI implementation and timing

Implement the confirmed **independent active-high CNV, no-busy RAC** sequence inside the new device dependency. GPIO8 remains the CNV wire; “CE0” identifies its alternate pin function, not its intended owner. Use GPIO8/pin 24 -> CNV, GPIO11/pin 23 -> SCK, GPIO10/pin 19 (MOSI) -> DIN, and GPIO9/pin 21 (MISO) <- SDO, with common ground, VDD/VIO approximately 3.3 V, COM grounded, and external REF approximately 3.32 V. The observed REFIN voltage does not change the confirmed REF=110 selection.

Use the libgpiod >=2.0 **C API** from C++, together with direct spidev. This mirrors the scripts' capabilities without adding a Python runtime, a second scheduling loop, or a new common GPIO/SPI framework. The MCP3208 lgpio wrapper is unsuitable unchanged: its open implementation masks SPI flags to the mode bits and cannot preserve `SPI_NO_CS` or own independent CNV timing.

Proposed resource/setup sequence:

1. Use the reported `/dev/gpiochip0` offset 8 as the initial configurable profile; validate identity/availability on the target before use. The user has observed this line owned by `spi0 CS0`, so treat a busy line as an expected actionable startup failure until the user releases that ownership. Request one persistent push-pull output, initial low, active-low disabled, with an application-specific consumer label. Check all allocations/settings/request results. Never request/release the line for each sample or assume `dtparam=spi=on` makes it available.
2. Open `/dev/spidev<bus>.<device>`, set and check mode 0 plus `SPI_NO_CS` using `SPI_IOC_WR_MODE32`/readback, eight bits per word, MSB first, and the configured clock. Keep the GPIO request valid throughout. Check mode setup failures and unwind the GPIO request/chip/SPI descriptor as applicable.
3. Confirm through user-run measurement that SPI setup/transfers cannot drive GPIO8. `SPI_NO_CS` suppresses chip-select operation; it does not itself remove a kernel line reservation or change pinmux. If GPIO8 is busy, report the owner/setup error and stop startup, without taking the line from another driver.
4. With the SPI pins quiet and CNV low, perform the controlled warmup/readback sequence. Maintain only one device owner, called by the acquisition thread after initialization; callbacks do not perform GPIO/SPI I/O.

The per-conversion primitive is:

```text
CNV low / previous word complete
  -> enforce prior acquisition, cycle-spacing, and SCK/DIN quiet deadlines
  -> timestamp software scan start before the first useful GPIO-high request
  -> drive CNV high: sample the channel already acquired
  -> after successful GPIO-high completion, measure a monotonic high-hold deadline
  -> keep CNV high and SCK/DIN quiet through conversion
  -> drive CNV low, then enforce first-bit setup
  -> transfer two bytes with SPI_NO_CS: read that conversion, write a future CFG
  -> decode all 16 bits, leave CNV low for the next acquisition/idle interval
```

Use bounded monotonic deadline waits for the short ADC guards; do not rely on Python overhead or an ordinary sleep achieving microsecond precision. Short waits may spin in the already configured acquisition thread and must be measured for CPU use. The outer timer/semaphore scheduling and publication remain those of `mcp3208Ctrl`. GPIO ioctls and scheduling introduce edge uncertainty; this approach provides explicit minimum timing, not a hardware clock or an upper jitter bound.

The low-voltage constraints are from [datasheet Table 6 and the digital-interface sections](https://www.analog.com/media/en/technical-documentation/data-sheets/ad7682_7689.pdf#page=8), for the B-grade part near 3.3 V and up to 85 C:

| Constraint | Required handling |
| --- | --- |
| `tCONV` | Allow the 3.2 us maximum before lowering CNV/reading. Start with the scripts' conservative **10 us high hold**, measured after a successful high-write returns. A separately qualified 4 us setting is an optimization candidate; reject values below the datasheet bound. |
| `tACQ`, `tCYC` | Full bandwidth requires at least 1.8 us acquisition after EOC and at least 5 us between rising edges. The acquisition interval includes time after EOC while CNV is still high plus the low/read/idle interval. Enforce all three constraints; do not impose an unnecessary second 1.8 us wait if it has already elapsed. |
| Unknown startup bandwidth | Before full-bandwidth CFG is known active, use a conservative >=20 us cycle and enough acquisition for a possible quarter-bandwidth state. Deriving 16.8 us acquisition from a 20 us cycle minus 3.2 us conversion is a conservative startup policy, not a separate quoted timing-table value. Do not accelerate flush calls based on incidental syscall overhead. |
| `tCNVH` | At least 10 ns; holding high through conversion exceeds this. CNV stays high at EOC, disabling busy indication. |
| CNV-low/first bit | At least 10 ns from CNV low to SCK rising; MSB enable can take 21 ns with VIO above 3 V, plus host setup margin. Initially enforce a conservative **1 us low-to-SPI guard** after the GPIO-low operation. |
| SCK/DIN/SDO | SCK high/low each at least 12 ns; SDO validity can lag falling SCK by 24 ns; DIN setup/hold each 5 ns. Check mode-0 half-cycle margins and actual SCK/loads, not an advertised maximum rate. |
| Aperture quiet window | Keep SCK/DIN quiet approximately 20 ns before and 10 ns after CNV rises. Initially enforce a **1 us post-SPI quiet guard** before the next high request. The high hold covers the following quiet interval. Scope MOSI idle transitions as well as SCK. |
| Conversion quiet window | No serial activity in the sensitive interval from `tDATA=1.2 us` to EOC. The proposed RAC primitive keeps serial pins quiet during the whole conversion. |

Normal readout uses 16 clocks, with CNV low throughout, because this is no-busy mode. Do not copy 17-clock/busy-mode framing, invert CNV because it sits on a CE pin, or let SPI toggle it. A 32-clock readback transaction is explicitly tracked during verification only.

**Pin ownership and alternatives:** the confirmed plan retains the wire and manual CNV. The known possible CS reservation is a target-setup prerequisite, not a reason to change the C++ protocol. If needed, document the smallest device-tree/pinmux adjustment releasing/reassigning that CS while retaining SPI0 data/clock; the user reviews/performs it during commissioning. Do not implement an automatic CE0 fallback or silently select another GPIO when a request fails. Rewiring or kernel/hardware-paced CNV would require a later proposal only if ownership/timing cannot be resolved within the agreed approach.

### 4. Configuration latency, initialization, and N-conversion scan schedule

Keep the following distinction explicit in code, documentation, and tests:

- Call k first raises CNV for conversion k, waits for EOC, then reads **conversion k**.
- The channel for conversion k was acquired before that rising edge. A CFG transmitted in the read phase of call k is applied at EOC of conversion k+1, for acquisition/conversion k+2.
- Consequently, CFG[k] controls result[k+2], even though the data read by call k comes from that call's own CNV edge. This agrees with the two-entry CFG history in the [ADI RAC reference implementation](https://github.com/analogdevicesinc/no-OS/blob/main/drivers/adc/ad7689/ad7689.c). It is not an instruction to delay an already valid raw sample by two frames.
- A zero-TX hold has CFG[13]=0; it retains settings rather than selecting IN0. It is suitable for a settled fixed channel, but does not implement a multichannel scan by itself.

At startup/reopen/reconfiguration, CFG and the first replies are unknown. Send the intended explicit settings using conservative startup timing; perform at least the required two dummy conversions and conservatively discard the first three replies plus any further reply with unproven provenance. Track the active acquisition channel and next pending CFG separately, with configuration/frame generations and validity. Do not substitute a fixed 20-call loop for establishing the states. Verify CFG through delayed readback, restore RB=1, and rebuild the normal full-bandwidth history before accepting output. Do not use the all-ones factory preload, which enables different reference/sequencer settings.

Let `C[0..N-1]` be the ordered physical inputs. Before the first frame, repeatedly configure C0 until both history stages are known C0, then perform a discarded warmup call that writes C1 (C0 for N=1). At that call's EOC the earlier C0 setting is active; its post-EOC write leaves C1 pending. The ready state is therefore **C0 acquired/acquiring, C1 pending**, with CNV low. The first useful rising edge occurs only after the frame's timer/semaphore trigger and delay.

For each useful conversion i within a frame, transmit `CFG(C[(i+2) mod N])` in its post-conversion read phase. Read the code for `C[i]` from the conversion initiated by that same call. An illustrative four-channel frame is:

| Call after the current frame trigger | Channel sampled/read | Post-EOC CFG transmitted | State prepared for the following call |
| --- | --- | --- | --- |
| 0 | C0 -> output slot 0 | C2 | Acquire C1, pending C2 |
| 1 | C1 -> output slot 1 | C3 | Acquire C2, pending C3 |
| 2 | C2 -> output slot 2 | C0 | Acquire C3, pending C0 |
| 3 | C3 -> output slot 3 | C1 | Acquire C0, pending C1; ready for the next frame |

For N=2, the transmitted CFG list is C0,C1; for N=1 it is C0. Use the general modulo schedule and test these boundary cases explicitly. An arbitrary-length idle wait with CNV low does not itself start another conversion, so no pre-trigger result/guard conversion is created at the frame boundary. The persistent two-slot history returns to the correct start state after each complete scan. **Normal steady acquisition uses N conversions and N two-byte transfers per N-sample frame**, with zero per-frame dummy conversions; only initialization/verification/recovery intentionally discards conversions.

Fill a staging buffer and publish only when all N results match the expected channels/generation and were sampled after the current frame trigger. A partial transfer, failed GPIO write, uncertain interrupted operation, channel-list change, or acquisition restart invalidates the pipeline and whole pending frame. Stop and re-prime; do not advance the schedule on a failed conversion or resume at a guessed channel. Verify this schedule independently using conversion-varying channel values/readback rather than a mock that returns the latest requested input.

### 5. Sampling budgets and rate admission

The scripts demonstrate the intended single-input method, not measured bounds for the production app. At 3.3 V, full bandwidth permits 200 kSPS and quarter bandwidth 50 kSPS aggregate. Eight channels at 8000 FPS need 64000 conversions/s, so the production schedule requires full bandwidth and no recurring dummy-per-channel loop.

Let H be the enforced high hold, S the low-to-SPI setup guard, Q the post-SPI quiet guard, and fSPI the actual SPI clock. Ignoring all syscall, scheduling, and publication costs, the proposed guarded N-call scan has the following budget:

```text
per_conversion >= H + S + 16/fSPI + Q
scan_budget    >= N * per_conversion
frame_period   >= scan_budget + GPIO/SPI/framework overhead + reserved margin
```

The final quiet guard can overlap a sufficiently long idle interval; the table conservatively budgets it per conversion, including at the frame boundary. Acquisition and rising-edge spacing are additional checks, with already elapsed intervals accounted for once. H is a hold deadline after the high-write returns; actual GPIO-write latency extends the hardware high interval. These are configuration-derived bounds, not claims that a requested delay or software timestamp measures the physical edge.

| Setting | Guarded time before GPIO/SPI/framework overhead | Implication for a 125 us frame |
| --- | --- | --- |
| Ideal 8 channels at 1 MHz, ignoring all ADC guards | 128 us of wire time alone | Impossible at 8000 FPS. |
| Scripts' 2 MHz and H=10 us, 8 channels, even omitting S/Q | 144 us | Cannot support eight channels at 8 kHz with this hold. |
| Production starting guards H=10 us, S=Q=1 us, 1 MHz, N=2 | 56 us | Useful low-rate bring-up profile; 8 kHz still requires measured overhead. |
| H=10 us, S=Q=1 us, 2 MHz, N=1 / N=2 / N=8 | 20 / 40 / 160 us | Single/pair targets may fit; eight-channel 8 kHz cannot. |
| H=10 us, S=Q=1 us, 4 MHz / 8 MHz, N=8 | 128 / 112 us | 4 MHz fails; at 8 MHz even the inherited 20 us synchronized cadence guard exhausts the remaining 13 us before overhead. |
| Candidate H=4 us, S=Q=1 us, 2 MHz / 4 MHz / 8 MHz, N=8 | 112 / 80 / 64 us | 2 MHz lacks conservative synchronized margin; 4/8 MHz merit qualification. These are not guarantees that GPIO/SPI syscall throughput fits. |

Keep the confirmed 1 MHz bring-up option and labeled 2 MHz profile matching the scripts' speed/10 us high hold. Evaluate 4/8 MHz and a 4 us hold through explicit settings and user-run timing qualification. Only admit a higher requested rate when its configured guard budget passes; only call the profile qualified after measuring GPIO/SPI cost and the agreed 60 s acceptance. The ADC's 3.2 us conversion maximum and 1.8 us acquisition/5 us cycle constraints remain mandatory. Start full-bandwidth warmup at the conservative >=20 us cycle until CFG is known active, regardless of the later speed. Record rates/jitter/skew for every enabled channel; no unrequested numerical jitter/skew threshold is a prerequisite to implementing these measurements.

There are at least two GPIO value-setting operations and one SPI ioctl per conversion; eight inputs at 8 kHz therefore imply approximately 192000 such operations/s, before clock reads, thread work, and ImageStreamIO publication. Benchmark this overhead explicitly. Do not propose a single contiguous SPI transfer/batch as a substitute for missing intermediate CNV pulses; manual CNV still needs one complete controlled conversion per input.

Keep the MCP3208 timer integrator and semaphore delay/cadence model, but bound interval/correction values and record misses. Do not copy the Python loop's unlimited immediate catch-up on an expired target. When late, count missed timer periods, avoid publishing stale/partial data or fabricated replacements, and schedule the next valid frame according to the documented timer policy. In semaphore mode report ADC overruns separately from producer-ID gaps; skip explicitly when a full valid scan cannot satisfy the agreed timing policy. Do not silently increase SPI speed, shrink timing/cadence guards, or claim mean rate proves bounded jitter.

If the target fails, measure GPIO high/low latency, SPI call time, guard waits, scheduling, and publication separately. Qualify a safe shorter hold and higher SCK first. If userspace overhead/jitter still prevents the agreed rate, present an approved kernel/hardware-paced acquisition option rather than adding a second unreviewed architecture to this app. Quarter bandwidth remains a possible single-channel comparison setting in a separate qualification harness, not the initial production operating mode.

### 6. Application integration, configuration, and recovery

- Preserve existing `framegrabber.*`, `telemeter.*`, `fps.*` source subscription, `numChannels.*` source subscription, and `synchro.*` keys. Register and load `accel.numChannels` properly. Add `accel.fps` as the explicit initial **frame/per-enabled-channel** rate; compute all timer targets consistently at load/start, rather than leaving constructor-derived periods unchanged after config.
- Add `accel.channels` as an optional ordered integer list. If omitted, retain prefix selection and `numChannels` control. If present, derive N; reject an explicitly conflicting count, and reject count-only runtime changes that cannot unambiguously preserve the explicit mapping. Expose the active mapping read-only and require reconfiguration for mapping changes.
- Final device/profile keys: `spi.bus=0`, `spi.channel=0`, `spi.baud=1000000`, `cnv.chip=/dev/gpiochip0`, `cnv.line=8`, `adc.conversionTime_ns=10000`, `adc.spiSetup_ns=1000`, `adc.quietTime_ns=1000`, and `adc.vref=3.32` using the reported approximate reference. Keep the chip/line/reference configurable for verified mapping and measured calibration. The example retains `accel.numChannels=2` and `accel.fps=2000` commissioning defaults and shows the explicit 8000 FPS goal, 2 MHz comparison profile, and qualified 4/8 MHz/4 us candidates. Mode 0, `SPI_NO_CS`, full bandwidth, REF=110, and sequencer-off are fixed initial operating settings; validate guard minima and combined rate admission. No `spi.batch` or automatic-CS switch is added.
- Use the distinct `ad7689Ctrl` application/configuration identity and `framegrabber.shmimName=accad7689_test` in the initial example. Preserve the N-by-1 `UINT16` frame and retained timestamp/counter/semaphore conventions of `acc000`. Do not inherit an `acc000` producer instance/name implicitly or activate a replacement service. Any later explicit production stream selection requires the separate site/handover work described below.
- Validate FPS/count/list/reference/clock requests before accepting current values; leave a rejected request in an appropriate INDI error state. Make finite checks effective despite the repository optimization defaults. Keep alpha clamping and signed synchronization-offset wrapping behavior.
- Stage runtime configuration changes under the existing application mutex convention (`{ //mutex scope`), and let the acquisition thread own ADC/GPIO state and apply changes at a stopped frame boundary. Resize sample/staging vectors, recreate stream geometry as necessary, clear timing/association state, and re-prime before resuming. Never close, release, or reconfigure SPI/GPIO concurrently with a scan, or hold the INDI mutex across an entire blocking wait/transfer. CNV mapping and electrical mode are startup configuration; changing either requires a controlled restart rather than a live pin-ownership change.
- Replace the two per-channel read loops with the same `readScan` path. Fill a separate staging buffer; commit to `m_values` only when every expected channel result passes generation/association checks. Use positive acquisition return codes for intentional no-frame cases and negative for a persistent critical failure; map all ADC errors into logging/recovery without uncaught device exceptions.
- Keep scan-start `CLOCK_REALTIME` acquisition timestamps and monotonic deadline/duration measurements separate. Set `m_currImageTimestamp` after the frame trigger/delay and immediately before the first useful CNV-high request in both modes, preserving a software scan-start convention and filling the timer-mode omission. Like the scripts' `perf_counter_ns()` marker, it precedes the physical GPIO edge by a variable call/scheduling delay. Record or measure GPIO-call bracketing and scan duration for qualification; do not label the timestamp as a hardware aperture measurement or shift it to SPI read completion. Preserve existing trigger/arrival/read-latency diagnostic names/units and explain their software-event meaning.
- Preserve local/publication and synchronization-producer counters and reset conventions. Add read-only ADC health diagnostics for GPIO/SPI errors, invalid frames, warmup/recovery discards, overruns/missed periods, recovery attempts, measured scan duration, and total/useful CNV events. Keep producer-ID gaps distinct from ADC failures and intentional warmup conversions; throttle repeated logs outside the high-rate path. Normal scans perform N useful conversions with no recurring guard conversion.
- Handle SPI/GPIO startup, ownership, or CFG mismatch as a reported startup error with partial-resource cleanup. On acquisition failure, stop publishing, invalidate pending pipeline tags, and enter bounded reconfiguration/reconnect attempts; persistent failure returns through the framegrabber error path. EINTR does not justify replaying an operation whose hardware progress is unknown. On shutdown, join acquisition first, then complete any known in-flight conversion interval before best-effort CNV-low cleanup, close SPI, release the line request/chip, and release synchronization resources. Do not add a cleanup rising edge; report GPIO cleanup failures and keep the destructor nonthrowing.

### 7. Build, documentation, and approved implementation sequence

1. After approval, work on the existing namespaced feature branch (currently `parkertjohnson1/ad7689Ctrl`) or an appropriately namespaced successor. Implement/test the new device protocol and transport first, then integrate the new app. Do not change the reference application.
2. Reuse standard app/test make includes and register the new Pi-targeted executable alongside existing ACC applications without replacing MCP3008. Standalone development uses `make -C apps/ad7689Ctrl`; a specific deployed host role/service is not needed to implement that target. Use pkg-config for libgpiod >=2.0 and gate generic builds/coverage on Linux headers/dependencies. Build the production object against real v2 headers while fake driver tests keep their stub/linkage private. Document dependency provisioning and version recording; no external setup/service repository file is modified in this producer implementation.
3. Add the example/README for the confirmed REF/GPIO/SPI profile, correct scaling, ordered channels, target/guard budgets, two-call CFG look-ahead, software `atime`, scan duration/CNV offset, startup discards, and the exact agreed performance reporting/acceptance. Include the `ad7689Ctrl` identity, `accad7689_test` stream, GPIO ownership/electrical prerequisites, and explicit deferred consumer/site handoff. Keep scratch scripts unchanged and use standard MagAO-X `-n`/framegrabber configuration conventions.
4. Run the two new offline suites and unchanged MCP3208 regressions on a suitable local Linux environment; build the new executable. Commands include `make -C apps/ad7689Ctrl`, `make -C tests -f Makefile.one t=../apps/ad7689Ctrl/tests/ad7689Ctrl_test`, and the analogous ADC/MCP3208 targets. Exercise startup/stop with fake GPIO/SPI, never by opening real instrument devices as a CLI check. `accelFFCtrl` source/tests stay outside this delivery; its companion owns the full-range/calibration/admission tests and later end-to-end qualification.
5. Perform a full documentation pass on every touched source/header/test file, including inline declaration parameter docs, `m_` state names, data-before-accessor named sections, file-level `\file`/`\brief`, and removal of copied `\author`. Keep nontrivial definitions outside class declarations. Run repository-configured clang-format on all touched C/C++ files; formatting Markdown with clang-format is not applicable.
6. Keep functional, documentation-only, and any formatting-only cleanup in clean commits if commits are requested; include the current plan with associated engineering changes and follow required model attribution. Prepare any requested PR title/description with AGENTS.md attribution. No commit, PR, install, or hardware activation is part of this planning task.
7. Hand build/capture instructions to the user for board qualification after the source/offline checks. Record actual versions, GPIO ownership fix, electrical checks, accuracy and 60 s timing results. Resolve any required pinmux/system change with the user before it is applied. After independent hardware/acquisition qualification, hand the documented producer contract/results to the separate consumer and production-deployment work; identify and review actual site files before changing them. No active `acc000` producer is replaced automatically.

## Implementation Readiness

**Ready for approval. No critical unresolved design, scope, or information blocker prevents implementing the standalone `ad7689Ctrl` source, tests, and in-tree build/example/documentation changes.** All five answers are incorporated. This readiness statement is not approval to start implementation, a claim of hardware qualification, or permission to change a running producer.

| Remaining item | Stage and required handling | Blocks source/offline implementation? |
| --- | --- | --- |
| GPIO8 previously reserved by `spi0 CS0` | Before a hardware run, the user must verify the reported chip/offset and release SPI CS ownership while retaining manual GPIO8. `SPI_NO_CS` alone is insufficient; implement/report clean EBUSY startup failure and document the prerequisite. Do not steal the line or rewire automatically. | No; it can block startup on the target until resolved. |
| REF/REFIN source/routing, filtering, and voltage tolerances | Before electrical acceptance, check the schematic and measured operating rails/reference. Use the confirmed REF=110/direct-REF selection in software; the approximately 3.32 V REFIN measurement does not select the internal buffer. The REF/VDD relationship remains a hardware qualification check against the cited datasheet. | No; the required software mode is settled. |
| Exact OS/kernel/libgpiod/spidev versions not recorded | Record versions/device tree during commissioning, require libgpiod >=2.0 for the C++ build, and fail dependency checks clearly. No Python runtime is required by the app. | No; an actual missing development dependency would need provisioning before a production build. |
| Physical timing traces and sustained 8 kHz capability | User-run commissioning must verify CNV/SCK timing, timestamp offset, and each claimed channel-count/profile against the 1%/60 s/zero-unexplained-loss criteria. Jitter/skew are measured and reported without an invented numerical limit. Unsupported profiles remain rejected/unqualified. | No; measurement is required before claiming target performance. |
| `accelFFCtrl` full-range/calibration/geometry support | Separate companion after independent producer testing. Retain raw 16-bit samples and current stream conventions here; record the known 4095 ceiling/two-element reader and required handoff. | No for this producer; yes for later full-range consumer handover. |
| Actual site configuration/service files and production handover | The user has deferred production deployment. Identify/review those files before implementing site changes, use a distinct test instance/stream now, and coordinate exclusive `acc000` ownership later. No host/service path is guessed in this source implementation. | No for in-repository work; required before production site changes. |

The only outstanding authorization for the scoped source implementation is the user's approval of this finalized plan. Hardware/system changes and the deferred companion/deployment tasks retain their separate review boundaries.

## Follow-up and Edge Cases

- **Implementation approval remains pending; all five answers are resolved.** Manual GPIO8 CNV/SPI_NO_CS, REF=110 with the reported rails, ordered multichannel frames/software timestamps, performance acceptance, and companion/deployment scope are settled. Remaining target checks are staged in Implementation Readiness and do not require reopening these design choices before coding.
- **Raw compatibility versus calibration:** Never downshift AD7689 samples to fit a 4095 ceiling. A full-scale 16-bit midpoint is near 32768, but actual accelerometer zero is measured; `VREF/65536` is ADC-pin volts/count, not automatically the empirical end-to-end volts/count used by `accelFFCtrl` or `accelSim`.
- **Analog settling/aliasing:** Minimum ADC acquisition time does not prove the custom input amplifiers, impedance, and filters settle to 16-bit accuracy after mux switching. Confirm via distinct-level/full-scale-step tests. The ADC's full-bandwidth filter is not an 8 kHz anti-alias filter; assess the actual PCB filter separately without redesigning it in this task.
- **Sampling skew:** An N-channel frame contains sequential apertures. Report first-to-last skew at each setting, and retain output order; simultaneous sampling would require a different hardware/processing design. A single frame `atime` cannot represent each channel's aperture exactly.
- **Timing/restarts:** Stop/reconfiguration during warmup or a scan, synchronization-stream recreation, stale semaphore backlogs, producer-counter resets, realtime clock steps, and rate-source changes must not create a valid-looking old frame. Keep monotonic timing for deadlines, reset association generations, and test affected counters/timestamps.
- **Limits of detection:** The ADC provides no sample CRC or device ID in this interface. A correctly sized SPI ioctl does not prove analog correctness or even board presence; readback and electrical measurements improve detection, while a constant 0/65535 sample alone proves neither a fault nor valid hardware.
- **8 kHz status:** The full-bandwidth ADC has capacity for 64000 useful conversions/s with the N-call schedule, but 1 MHz/eight-channel operation and the scripts' 2 MHz/10 us eight-channel profile cannot meet 8 kHz. Linux GPIO/SPI/publication overhead and edge jitter remain unqualified. If bounded software changes cannot meet the agreed requirement, return for approval of hardware-paced CNV/kernel/DMA acquisition instead of changing the frame/trigger architecture silently.
- **Script analysis limits:** Do not treat printed filenames as exported data, run labels as channel/rate truth, or a Welch plot based on mean sample spacing as proof of a uniformly sampled record. An independent signal/logic-analyzer comparison and saved raw/timestamp/configuration records are required to qualify accuracy and timing. Correct future voltage analysis to `/65536`; no Python plotting code is incorporated into the production acquisition path.
- **Planning verification:** Preserve the original bytes before `# Agent Findings and Plan`, check Markdown whitespace/links and calculations, and confirm that this plan is the only file changed by this task. Existing user changes in `agents/task_template.md` are unrelated and must remain untouched. No application tests or hardware tests can validate an implementation that has not yet been authorized or written.

Planning checks completed for the answer-review revision: the 13,372-byte protected prefix matches the start-of-review snapshot (SHA-256 `a772f9d8e7eee994e1fbfa886ef9138a9af48f9c5f13f7a38a22b10f689d2766`), and all five complete original question/answer blocks are byte-for-byte unchanged. Confirmed hardware/settings, performance criteria, timestamp/stream behavior, scope/file list, tests, and staged readiness were checked for consistency; Markdown tables/fences and changed-text whitespace checks passed. Scratch/reference sources, AGENTS.md, and the unrelated task template match their baseline hashes. The prior static script/CFG/scan/budget checks remain applicable. Application builds/tests, script execution, and hardware validation have not been run; the finalized plan is ready for approval, and implementation has not started.

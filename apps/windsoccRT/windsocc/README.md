# WindsoCC

Post-observation analysis of turbulence using MagAO-X camwfs images.

## Installation

This project uses `uv` for dependency management. To install:

```bash
uv sync
```

To install the package in editable mode:

```bash
uv pip install -e .
```

From the parent `apps/windsoccRT/` directory you can also run `make install-windsocc-python` (see the `Makefile` there) to install into the configured MagAO-X Python environment.

## Command-Line Tools

The following entry points are available after installation:

- `ws_partition` - Organize FITS files into subdirectories based on time intervals
- `ws_reduce` - WindsoCC data reduction pipeline
- `ws_xcorr` - Cross-correlation analysis
- `ws_distill` - Background subtraction, bias correction, noise reduction
- `ws_measure` - Wind measurement and analysis
- `ws_realtime` - Run realtime WindsoCC processing in Python (offline FITS, custom reader, or live ``magaox.shmim``)
- `ws_debug_imports` - Step through WindsoCC imports to isolate the first crashing dependency

## Project Structure

```
windsocc/
├── src/windsocc/          # Main package source code
│   ├── analysis/          # Analysis modules
│   ├── io/                # Input/output handling
│   ├── preprocessing/     # Data preprocessing
│   └── visualization/     # Visualization tools
├── scripts/               # Standalone utility scripts
├── tests/                 # Test suite
└── pyproject.toml         # Project configuration
```

## Development

This project follows modern Python packaging standards with:
- `src/` layout for source code
- `uv` for dependency management
- Entry points for CLI tools
- Standard project structure

## Debugging import and stream issues

Use `ws_debug_imports` to isolate import-time failures in your Python environment:

```bash
ws_debug_imports
ws_debug_imports --start-at polars
ws_debug_imports --direct-module windsocc.realtime
```

To exercise live shmim acquisition on the RTC, use `ws_realtime` with `--source-type shmim`:

```bash
ws_realtime --source-type shmim --stream-name aol1_imWFS2 --frame-count 512 --config /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-python-stream
```

This uses `magaox.shmim.Image` directly, so the MagAO-X Python package and `ImageStreamIOWrap` must be importable in that environment. The batch is processed with the same in-memory pipeline as other `ws_realtime` modes (`process_collected_batch`).

Historical MagAO-X C++ embedded-Python notes (`windsoccRT`, probe binaries) are preserved under [`../archive/README.md`](../archive/README.md).

## windsoccRT standby and readiness (RTC)

The `windsoccRT` INDI driver (`xapp/windsoccRT`) runs one WindsoCC batch per `loop()` only when:

1. The configured shmim file exists (`/milk/shm/{stream_name}.im.shm`). If missing, it logs **ERROR** at most once per 60 seconds (configurable via `shm_missing_log_interval_sec`).
2. Instrument readiness passes (when `enable_readiness_gating = true`). The pipeline is **blocked if any** of these is true:

| Blocker | Default INDI key | Condition |
|---------|------------------|-----------|
| Lab mode | `tcsi.labMode.toggle` | ON |
| Tel-sim in beam | `fwtelsim.filterName.in` | ON |
| WFS shutter closed | `camwfs.shutter.toggle` | ON (`shutter_closed_is_toggle_on = true`) |
| HO loop open | `holoop.loop_state.toggle` | OFF (ON = closed loop) |

On the first readiness failure after a successful pass, the driver logs one **WARNING** listing reasons, then suppresses further readiness warnings until readiness passes again. Between failures it sleeps with backoff: 1, 5, 10, 30, 60, 120, then 300 seconds (max).

The read-only INDI property `pipeline.state` reports `missing_stream`, `standby`, or `active`.

Example `/opt/MagAOX/config/windsocc.conf` overrides:

```toml
sleep_interval_sec = 0.0
enable_readiness_gating = true
shm_missing_log_interval_sec = 60.0
stream_name = "aol1_imWFS2"
```

Verify INDI state on the RTC:

```bash
indi_getprop tcsi.labMode fwtelsim.filterName camwfs.shutter holoop.loop_state windsocc.pipeline
```

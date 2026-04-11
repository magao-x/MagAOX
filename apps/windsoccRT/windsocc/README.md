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

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

## Command-Line Tools

The following entry points are available after installation:

- `ws_partition` - Organize FITS files into subdirectories based on time intervals
- `ws_reduce` - WindsoCC data reduction pipeline
- `ws_xcorr` - Cross-correlation analysis
- `ws_distill` - Background subtraction, bias correction, noise reduction
- `ws_measure` - Wind measurement and analysis
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

## Debugging Import Crashes

Use `ws_debug_imports` to isolate import-time failures in the same Python environment used by `windsoccRT`.

Examples:

```bash
ws_debug_imports
ws_debug_imports --start-at polars
ws_debug_imports --direct-module windsocc.realtime
```

For embedded-Python tests, build and run `windsoccImportProbe` from `apps/windsoccRT/` (or `/opt/MagAOX/bin/` after `make install`) with the same `windsocc.pythonImportRoot` used by the app:

```bash
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --stepwise
```

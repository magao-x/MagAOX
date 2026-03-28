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

Recommended follow-up matrix:

```bash
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --stepwise
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --spawn-thread
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --spawn-thread --install-signal-handlers
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --report-ids
```

Interpretation:

- If baseline succeeds but `--spawn-thread` fails, a pre-existing thread is the leading suspect.
- If baseline succeeds and only `--install-signal-handlers` changes behavior, signal context matters.
- If all probe modes succeed but `windsoccRT` still fails, the remaining gap is likely deeper `MagAOXApp` context (especially the real log thread or privilege transitions).
- Use `--report-ids` to compare the probe's real/effective ids against the installed `windsoccRT` process context before investigating setuid-related behavior further.

For in-app A/B testing, compare `windsoccRT` with and without `windsocc.importBeforeShmim=true` while leaving the other debug flags enabled.

To isolate the full `MagAOXApp` lifecycle without `shmimMonitor`, build and run `windsoccAppImportProbe` from `apps/windsoccRT/` (or `/opt/MagAOX/bin/` after `make install`):

```bash
./windsoccAppImportProbe -n windsoccAppProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
```

Recommended comparison flow:

```bash
ws_debug_imports
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
./windsoccImportProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --spawn-thread --install-signal-handlers
./windsoccAppImportProbe -n windsoccAppProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
/opt/MagAOX/bin/windsoccRT -n windsocc --windsocc.importBeforeShmim=true ...
```

Interpretation:

- If only `windsoccAppImportProbe` fails, the remaining suspect is the true `MagAOXApp` lifecycle or deployment context.
- If `windsoccAppImportProbe` succeeds but `windsoccRT` fails, the crash still depends on `windsoccRT`-specific state beyond the base app setup.

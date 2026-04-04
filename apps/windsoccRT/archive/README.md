# WindsoCC (archived README)

This file preserves the **previous** `windsocc/README.md` in full, including MagAO-X C++ embedded-Python troubleshooting (`windsoccRT`, probe binaries). The C++ sources and the legacy `Makefile` that built them are stored in this `archive/` directory. For current Python-only documentation, see [`../windsocc/README.md`](../windsocc/README.md).

---

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
- `ws_realtime` - Run one realtime-style WindsoCC batch in-process
- `ws_debug_imports` - Step through WindsoCC imports to isolate the first crashing dependency
- `ws_debug_stream_grab` - Grab a live MagAO-X shmim stream in Python and hand it directly to WindsoCC

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

To isolate the first threaded `run_embedded_batch_buffer` call without shmim or `MagAOXApp`, build and run `windsoccBatchCallProbe` from `apps/windsoccRT/` (or `/opt/MagAOX/bin/` after `make install`):

```bash
./windsoccBatchCallProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --config-path /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-batch-probe
./windsoccBatchCallProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --config-path /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-batch-probe --spawn-thread
```

Interpretation:

- If the single-threaded batch-call probe fails, the remaining issue is in the concrete Python batch-call ABI or callable execution itself.
- If the single-threaded batch-call probe succeeds but `--spawn-thread` fails, the remaining issue is likely the first threaded `PyGILState_Ensure()` / `PyObject_Call()` path.
- If both batch-call probe modes succeed but `windsoccRT` still fails, the remaining difference is likely live shmim/runtime sequencing inside `windsoccRT`.

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
./windsoccShmimImportProbe -n windsoccShmimProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
./windsoccBatchCallProbe --python-import-root /opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --config-path /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-batch-probe --spawn-thread
/opt/conda/envs/xpy3_13/bin/ws_debug_stream_grab --stream-name aol1_imWFS2 --frame-count 512 --config /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-python-stream
/opt/MagAOX/bin/windsoccRT -n windsocc --windsocc.importBeforeShmim=true ...
```

Interpretation:

- If only `windsoccAppImportProbe` fails, the remaining suspect is the true `MagAOXApp` lifecycle or deployment context.
- If `windsoccAppImportProbe` succeeds but `windsoccRT` fails, the crash still depends on `windsoccRT`-specific state beyond the base app setup.

To isolate `shmimMonitor` inheritance and config/load behavior without starting the shmim thread, run `windsoccShmimImportProbe` from `apps/windsoccRT/` (or `/opt/MagAOX/bin/` after `make install`):

```bash
./windsoccShmimImportProbe -n windsoccShmimProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src
```

To mirror more of `windsoccRT` after import, the shmim probe now supports callable resolution and an optional `PyEval_SaveThread()` handoff:

```bash
./windsoccShmimImportProbe -n windsoccShmimProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --windsocc.pythonCallable=run_embedded_batch_buffer
./windsoccShmimImportProbe -n windsoccShmimProbe --windsocc.pythonImportRoot=/opt/MagAOX/source/MagAOX/apps/windsoccRT/windsocc/src --windsocc.pythonCallable=run_embedded_batch_buffer --windsocc.saveThread=true
```

Interpret the two passes in order:

- First run the callable-resolution pass with the default `windsocc.resolveCallable=true` and `windsocc.saveThread=false`.
- Then enable `windsocc.saveThread=true` only if the callable-resolution pass succeeds cleanly.
- If the first pass fails, the remaining suspect is callable lookup or `PyCallable_Check()` rather than bare module import.
- If the first pass succeeds but the second fails, the remaining suspect is `PyEval_SaveThread()` / interpreter thread-state handoff.
- If both passes succeed but `windsoccRT` still fails, the next suspect is later worker-thread behavior such as the first `PyEval_RestoreThread()` and actual batch invocation path.

Additional interpretation:

- If only `windsoccShmimImportProbe` fails, `shmimMonitor` inheritance or its config/load path is the leading suspect.
- If `windsoccShmimImportProbe` succeeds but `windsoccRT` still fails, the remaining culprit is likely in `windsoccRT`-specific state or startup sequencing beyond the mixin.

To test live image acquisition entirely in Python, run `ws_debug_stream_grab` in the same environment as `windsoccRT`:

```bash
ws_debug_stream_grab --stream-name aol1_imWFS2 --frame-count 512 --config /opt/MagAOX/source/MagAOX/apps/windsoccRT/ws_config.yaml --output-root /tmp/windsocc-python-stream
```

Notes:

- `ws_debug_stream_grab` uses `magaox.shmim.Image` directly, so it depends on the MagAO-X Python package and `ImageStreamIOWrap` being importable in that environment.
- The script intentionally keeps handoff simple: it collects a live float32 cube in Python and then calls `windsocc.realtime.run_embedded_batch(...)`.
- If this succeeds while `windsoccRT` still fails, the strongest remaining suspect is the C++ embedding/threading path rather than the Python pipeline itself.

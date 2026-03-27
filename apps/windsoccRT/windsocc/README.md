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

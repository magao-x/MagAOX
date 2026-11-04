xrif2fits
=========

[TOC]

------------------------------------------------------------------------

# NAME

xrif2fits - convert MagAO-X `.xrif` image archives to FITS images or cubes.

# SYNOPSIS

```
xrif2fits [options]
xrif2fits -f file.xrif -D output-dir -t telemetry-dir[,telemetry-dir...] -l log-dir[,log-dir...]
xrif2fits -d input-dir -D output-dir -O
```

# DESCRIPTION

`xrif2fits` decodes MagAO-X `.xrif` image archives and writes FITS files.  In the
default mode it writes one FITS file per frame.  It can also write a FITS cube,
write metadata only, or report only archive time spans.

Each `.xrif` archive contains image data and frame timing data.  For each frame,
`xrif2fits` reads the acquisition time from the archive timing block.  When FITS
headers are enabled, it also reads camera and telemetry logs to fill in header
cards such as exposure time, camera state, telescope state, and derived metadata
configured for the camera.

Metadata extraction is deliberately conservative.  If a metadata value cannot be
established from available telemetry, the default behavior is to write
`NOT AVAILABLE`, report a recoverable error, and continue producing FITS files.
Use `--strict` when any metadata or recoverable log error should stop the run
before FITS files are written.

# OPTIONS

| Short | Long | Config-file key | Type | Description |
| --- | --- | --- | --- | --- |
| | `--camera` | `camera` | string | Camera/device name.  If omitted, this is inferred from the input filename.  Also sets the default header config to `<camera>_header.conf`. |
| | `--header.camera` | `header.camera` | string | Header configuration file.  Overrides the default selected from `camera`. |
| `-N` | `--noHeader` | `noHeader` | bool | Do not generate camera metadata headers. |
| `-d` | `--dir` | `dir` | string | Directory to search for input `.xrif` files, or directory prepended to names supplied with `-f`. |
| `-f` | `--files` | `files` | vector string | Input file or files.  If omitted, all `.xrif` files in `dir` are used. |
| `-D` | `--outDir` | `outDir` | string | Output directory.  Default is `./fits/`. |
| `-O` | `--overwrite` | `overwrite` | bool | Allow use of an existing output directory. |
| `-l` | `--logdir` | `logdir` | vector string | Base directories for `.binlog` log files. |
| `-t` | `--teldir` | `teldir` | vector string | Base directories for `.bintel` telemetry files. |
| | `--metaOnly` | `metaOnly` | bool | Write metadata output without decoding and writing images. |
| `-T` | `--time` | `time` | bool | Print one line per input archive: filename, start time, end time, total exposure time, and number of frames. |
| | `--noMeta` | `noMeta` | bool | Do not write `meta_data.txt`.  FITS headers are still written unless `-N` is also used. |
| `-C` | `--cubeMode` | `cubeMode` | bool | Write each archive as one FITS cube with a minimal header. |
| | `--strict` | `strict` | bool | Treat recoverable metadata/log errors as fatal before writing FITS files. |
| | `--show-details` | `showDetails` | bool | Print xrif image and timing compression details while processing. |
| | `--quiet` | `quiet` | bool | Suppress non-error status output. |
| | `--maxMetadataGap` | `maxMetadataGap` | double | Maximum allowed telemetry gap in seconds for metadata coverage.  Default is `25`.  Set negative to disable gap checks. |

# INPUT AND OUTPUT

Input files must use the standard MagAO-X filename convention so the camera name
and archive timestamp can be parsed.  All input files in one run must be from the
same camera.

In normal mode, output filenames are based on the frame acquisition timestamp:

```
<camera>_<timestamp>.fits
```

When `--noMeta` is not set, `xrif2fits` also writes `meta_data.txt` in the
output directory.  This file contains one line per frame with timing fields,
exposure time, and the configured metadata values.  Values that could not be
established are written as `NOT AVAILABLE`.

In cube mode, one FITS file is written for each archive:

```
<archive-stem>.fits
```

# HEADER CONFIGURATION

Header metadata is configured by a camera header configuration file.  If no
header config is specified, the default is:

```
<camera>_header.conf
```

The config search path normally comes from `$MagAOX_PATH/$MagAOX_CONFIG`.
Set `XRIF2FITS_CONFIGPATH` to override that search path for `xrif2fits`.

Header configuration files may include other config files:

```
include = common_telescope_header.conf
```

Device sections list flatlog event codes and member names to extract:

```
[tcsi]
telem_teldata = az
telem_teldata = zd
telem_teldata = pa

[camsci1]
telem_stdcam = emgain
telem_stdcam = fps
```

For each configured item, `xrif2fits` resolves:

- the device name, such as `tcsi` or `camsci1`;
- the flatlog event type, such as `telem_teldata`;
- the member name, such as `pa`;
- the FITS keyword, value type, and time behavior provided by that log type's
  accessor.

For example, `telem_teldata.pa` resolves to the FITS keyword `PARANG`.

# METADATA EXTRACTION

Metadata is looked up from telemetry files (`.bintel`).  `xrif2fits` builds a
file map for every device required by the camera header config and for the camera
itself.  The camera telemetry is always used to determine exposure time.

Each frame has an acquisition time from the `.xrif` timing data.  Exposure start
time is computed from the acquisition time and the nearest prior camera
`telem_stdcam.exptime` telemetry record.  Metadata values then use the exposure
start time, acquisition time, or midpoint depending on the metadata type.

## State Values

State metadata represents values that should remain true until changed, such as
camera mode, filter wheel state, or loop state.  `xrif2fits` finds the nearest
verified telemetry record before the exposure start and checks that the state is
covered through the exposure.  If a later record occurs during the exposure, the
value may be updated as long as the telemetry gap tolerance is satisfied.

If the required prior state is missing, unverifiable, or outside the metadata gap
tolerance, the FITS value is written as `NOT AVAILABLE`.

## Continuous Values

Continuous metadata represents values that can be interpolated, such as
temperatures or telescope coordinates.  `xrif2fits` finds verified telemetry
records bracketing the exposure midpoint and linearly interpolates to the
midpoint.

Both sides of the bracket must satisfy the metadata gap tolerance.  This prevents
old telemetry from silently being stretched across long gaps.

## Continuous Angle Values

Some continuous values are angles and must wrap correctly.  `PARANG`, from
`telem_teldata.pa`, uses angle-aware interpolation.  This means:

- `359 -> 1` interpolates through `0`, not through `180`;
- `1 -> 359` interpolates through `0` in the opposite direction;
- `-179 -> 179` and `179 -> -179` interpolate across the signed-angle wrap.

The interpolation uses the shortest angular delta in degrees and normalizes the
result using the apparent convention of the input samples.

## Telemetry Gap Tolerance

The default metadata gap tolerance is 25 seconds.  This is intended to allow
approximately one missed telemetry record when telemeters are expected to write
at least every 10 seconds.  Increase the tolerance with `--maxMetadataGap` when
you need to process older data with known long telemetry gaps.  Set it negative
to disable gap enforcement.

When a metadata value fails because of the gap tolerance, the warning names that
reason, for example:

```
xrif2fits: Metadata fwsci1 POS is NOT AVAILABLE due to gap in telemetry exceeding 25 sec.
```

# LOG CORRUPTION HANDLING

Binary telemetry/log files can contain corrupt entries.  `xrif2fits` treats some
log corruption as recoverable.  During file loading it follows the normal
length-chain traversal.  If traversal reaches an invalid entry, it byte-scans
forward to resynchronize on a plausible later entry.

The resync warning reports the byte span from the failed traversal point to the
resync point:

```
xrif2fits: Invalid log entry skipped while loading log file: source=/path/file.bintel sourceByte=8155228 resyncByte=8155292 (64 byte resync span; corrupt section may begin earlier)
```

That span is not necessarily the same as the total corrupt section that
`logsurgeon` would report.  `logsurgeon` verifies flatbuffer payloads byte by
byte and may identify an earlier corrupt candidate whose header was still
plausible enough for length-chain traversal.

When a telemetry entry has a sane envelope but fails flatbuffer verification
during metadata extraction, `xrif2fits` skips it, reports the source file, byte
offset, nearby timestamps, and continues searching for a verified entry.

# ERROR HANDLING

`xrif2fits` distinguishes fatal errors from recoverable metadata/log errors.

Recoverable errors include:

- metadata source directories not configured;
- no telemetry files for a requested metadata device;
- missing prior or following telemetry;
- telemetry gaps larger than `--maxMetadataGap`;
- corrupt log entries that can be skipped or resynchronized;
- metadata entries that fail flatbuffer verification.

In the default mode, recoverable errors are reported and the run continues.  FITS
headers and `meta_data.txt` use `NOT AVAILABLE` for affected values.

With `--strict`, any recoverable error stops processing before FITS files are
written.  Use strict mode when the output must not contain unavailable or
partially recovered metadata.

Fatal errors include:

- unreadable or malformed `.xrif` input;
- unsupported image data type;
- output directory conflicts when `-O` is not used;
- missing configured source directories when an explicit source was requested;
- header config parse failures.

# EXIT STATUS

`xrif2fits` returns:

| Code | Meaning |
| --- | --- |
| `0` | Completed with no recoverable errors. |
| `1` | Completed and wrote output, but one or more recoverable errors occurred. |
| `2` or greater | Fatal error.  No valid FITS output should be assumed. |

# EXAMPLES

Convert one archive, overwriting or reusing the output directory, and search the
standard telemetry/log locations:

```
xrif2fits -O \
  -f /srv/icc/opt/MagAOX/rawimages/camsci1/2026_05_11/camsci1_20260511035851088464615.xrif \
  -D /data/reduced/camsci1 \
  -t /opt/MagAOX/telem,/srv/icc/opt/MagAOX/telem,/srv/rtc/opt/MagAOX/telem \
  -l /opt/MagAOX/logs,/srv/icc/opt/MagAOX/logs,/srv/rtc/opt/MagAOX/logs
```

Convert every `.xrif` file in a directory:

```
xrif2fits -O -d /srv/icc/opt/MagAOX/rawimages/camsci1/2026_05_11 -D ./fits
```

Run in strict mode:

```
xrif2fits -O -f camsci1_20260511035851088464615.xrif -D ./fits --strict
```

Allow a larger telemetry gap for older data:

```
xrif2fits -O -f camsci1_20260511035851088464615.xrif -D ./fits --maxMetadataGap=7200
```

Print compression details while processing:

```
xrif2fits -O -f camsci1_20260511035851088464615.xrif -D ./fits --show-details
```

Suppress non-error status output:

```
xrif2fits -O -f camsci1_20260511035851088464615.xrif -D ./fits --quiet
```

Disable header metadata completely:

```
xrif2fits -N -f camsci1_20260511035851088464615.xrif -D ./fits
```

Print archive time coverage without writing FITS files:

```
xrif2fits -T -d /srv/icc/opt/MagAOX/rawimages/camsci1/2026_05_11
```

# SEE ALSO

`logdump`, `teldump`, `logsurgeon`, `xrif2shmim`

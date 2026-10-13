# Prompt
In utils/xrif2fits, we need to change the way error handling works.  Currently if it can't find a log or telem file with matching dates, it is a fatal error.  We instead want the fatal error to occur only if the top-level directory for that log/telem source doesn't exist.  E.g. if it is supposed to find `fwsci1` and there is no directory named `fwsci1`.  If the directory exists, but there are just no logs/telems in the date/time range, we want to notify the user but continue processing. The best solution would be to have the expected FITS header entries still be published, but with `NOT AVAILABLE` as the value.  

Review AGENTS.md then please analyze this problem, and then formulate a plan.  Update this document below with the plan, and do not make changes until I have approved it.  Do not modify this prompt above the "Plan" header below.  I have already created and switched to the feature branch for this.

# Plan

## Analysis

`utils/xrif2fits/xrif2fits.hpp` currently makes missing log/telemetry coverage fatal in
`xrif2fits::execute()` after it calls `logMap::loadAppToFileMap()` for each configured log and telem
source. The lower-level `logMap::loadAppToFileMap()` already has the desired split for one important
case: it returns `mx::error_t::dirnotfound` if the supplied source directory itself does not exist, but
returns `mx::error_t::noerror` when it searches the expected date span and finds no matching files.
`xrif2fits` erases that distinction by catching all exceptions during the load calls and then throwing
`xwcException` whenever `m_appToFileMap[app]` is empty.

The requested behavior should therefore be implemented in `xrif2fits`, not by making a broad
`logMap` behavior change. The app should:

- Treat a missing configured top-level source directory, or missing expected app subdirectory such as
  `<logdir>/fwsci1` or `<teldir>/fwsci1`, as fatal.
- Treat existing source/app directories with no matching date-range files as a warning and continue.
- Still emit FITS header cards for configured metadata keywords, using `NOT AVAILABLE` when the
  metadata cannot be read.

The `NOT AVAILABLE` behavior can be handled in the logger metadata helper layer as well as in
`xrif2fits`. That is acceptable for this work because this portion of `libMagAOX/logger` largely exists
to support `xrif2fits`, and this change does not require any flatlogs schema changes. `logMeta::card()`
already has the mechanics to write a string-valued fallback card when metadata lookup fails, but its
current sentinel is `"invalid"`, not the requested value.

One important edge case is `TELEM_STDCAM` exposure time. `writeImages()` currently uses exposure time to
compute `stime = atime - exptime`. Logger metadata lookups then use the exposure interval `[stime, atime]`:
state metadata searches for changes across the exposure, and continuous metadata interpolates at the
mid-exposure time. If exposure time is unavailable, the exposure window is not scientifically defined, so
the implementation should publish `NOT AVAILABLE` for interval-dependent metadata instead of pretending the
exposure is instantaneous.

## Implementation Plan

1. Add small xrif2fits-local availability tracking.
   - Track app names whose log and/or telem file maps are unavailable after scanning existing directories.
   - Add or expose a shared metadata fallback string, `NOT AVAILABLE`, without changing any flatlogs schema.
   - Keep changes in `utils/xrif2fits/xrif2fits.hpp` to match the app's header-only pattern.

2. Replace the fatal empty-map checks in `execute()`.
   - Before loading file maps for an app/source pair, explicitly verify the configured top-level source
     directory exists and that the expected app subdirectory exists.
   - If either required directory is missing, return/throw a fatal error with a path-specific message.
   - If directories exist but `loadAppToFileMap()` finds no files, print a warning and mark that app/source
     unavailable instead of throwing.
   - Stop swallowing all load exceptions silently; preserve the "missing date-range files is OK" path while
     surfacing real errors.

3. Make frame processing tolerant of unavailable telemetry.
   - Guard `m_tels.loadFiles()` in `execute()` so empty telemetry maps do not abort processing.
   - In the times-only and normal write paths, avoid calling `getPriorLog()` for an app whose telemetry map
     is unavailable.
   - When exposure time cannot be computed, warn once or sparingly and keep processing the frame, but do not
     perform interval-dependent metadata interpolation for that frame.

4. Publish fallback FITS cards and metadata text.
   - Update `writeImages()` so configured metadata cards are attempted independently of exposure-time success.
   - Update `logMeta` so failed metadata lookup renders as `NOT AVAILABLE` rather than `"invalid"`.
   - Add a helper for constructing an unavailable FITS card with the normal keyword/comment when `xrif2fits`
     knows in advance that no valid exposure interval exists.
   - For each configured `logMeta`, append a real card when telemetry is available, the exposure interval is
     valid, and lookup succeeds.
   - If telemetry/log metadata is unavailable, exposure time is unavailable, or lookup fails, append a string
     FITS card with the normal keyword/comment and value `NOT AVAILABLE`.
   - Write `NOT AVAILABLE` into `meta_data.txt` for those fields as well, including exposure time when it cannot
     be determined.

5. Add focused tests where feasible.
   - Inspect the existing xrif2fits test harness; if practical, add tests for missing source/app directory vs.
     existing empty app directory behavior.
   - If the current harness is only a placeholder and building realistic XRIF/log fixtures would be too large,
     document the test gap and at least build the xrif2fits utility.

6. Documentation and formatting.
   - Because `utils/xrif2fits/xrif2fits.hpp` will be touched, improve declaration/member documentation across the
     file according to `AGENTS.md`, keeping behavior changes minimal.
   - Run `clang-format` on touched C++ files.
   - Update this plan file with execution notes as implementation progresses, and keep it with the associated
     code changes if committed later.

## Open Questions / Assumptions

- I will interpret "top-level directory for that log/telem source doesn't exist" as both the configured base
  source directory and the expected app subdirectory being required. For example, if `-t /data/telem` is supplied
  and `fwsci1` metadata is needed, both `/data/telem` and `/data/telem/fwsci1` must exist.
- The current xrif2fits test file is a placeholder. I will add tests only if doing so can stay focused without
  building a large synthetic XRIF/log fixture system.

## Execution Notes

- Implemented directory validation in `xrif2fits` so configured source directories and expected app
  subdirectories are fatal when missing, while existing directories with no date-range files warn and continue.
- For multiple `-l`/`-t` roots, each configured source root must exist, but a given app/device subdirectory only
  has to exist under one of those roots. This supports local-plus-NFS source layouts.
- Empty source entries are ignored, and parser/default `.` entries are ignored when explicit source roots are
  present, preventing an implicit local root from making split source layouts fatal.
- Fixed the lower-level log-map fallback for cases where a previous log exists but no following date directory can
  be found; the previous log directory is now used instead of returning `dirnotfound` from a missing date subdir.
- Added logger metadata support for the shared unavailable value `NOT AVAILABLE` and a helper to construct
  unavailable FITS cards using the normal keyword/comment.
- Preserved exposure-time semantics: if `TELEM_STDCAM` exposure time is unavailable, xrif2fits does not invent
  an exposure interval for state/continuous interpolation and instead emits `NOT AVAILABLE` for dependent metadata.
- Replaced the placeholder xrif2fits test with focused tests for missing directories, empty existing directories,
  and unavailable FITS-card values.
- Started reducing chatty output by disabling `DEBUG_CRUMB` output unless `DEBUG` is defined, removing normal-run
  metadata lookup breadcrumbs from `xrif2fits`/logger code, and leaving real corruption/structural error messages in
  place.

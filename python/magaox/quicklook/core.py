#!/usr/bin/env python
'''lookyloo - convert xrif images + bintel telemetry into FITS

Example: Convert any camsci1 or camlowfs archives written
during an observation with 'lamVel' in the title recorded this year
and save the FITS files to folders under /home/jrmales/my_quicklook,
using a backup drive copy of the AOC/RTC/ICC data partitions.

    lookyloo --camera camsci1 --camera camlowfs \\
        --data-root /media/magao-x_2022a/aoc \\
        --data-root /media/magao-x_2022a/rtc \\
        --data-root /media/magao-x_2022a/icc \\
        --output-dir /home/jrmales/my_quicklook \\
        --title lamVel --partial-match-ok

Example: Convert camsci1 and camsci2 (the defaults) archives from any
observation in 2022 and record verbose output from lookyloo itself
to a log file in the current directory, saving outputs to folders in
the current directory as well.

    lookyloo --output-dir . --year 2022 --verbose
'''
import re
import subprocess
import pathlib
import typing
import dataclasses
import os
import time
import datetime
from datetime import timezone
from concurrent import futures
import logging
import glob
import tempfile
from astropy.io import fits
import shutil

from ..utils import parse_iso_datetime, format_timestamp_for_filename, utcnow
from ..constants import (
    PRETTY_MODIFIED_TIME_FORMAT, LINE_FORMAT_REGEX, LINE_BUFFERED, MODIFIED_TIME_FORMAT, OBSERVERS_DEVICE, HISTORY_FILENAME, FAILED_HISTORY_FILENAME, XRIF2FITS_TIMEOUT_SEC, SLEEP_FOR_TELEMS, ALL_CAMERAS, DEFAULT_SEPARATE, DEFAULT_CUBE, CHECK_INTERVAL_SEC, LOOKYLOO_DATA_ROOTS, QUICKLOOK_PATH,
    FOLDER_TIMESTAMP_FORMAT,
)

log = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True, eq=True)
class TimestampedFile:
    path : pathlib.Path
    timestamp : datetime.datetime

    def __str__(self):
        return f"<TimestampedFile: {self.path} at {self.timestamp.strftime(PRETTY_MODIFIED_TIME_FORMAT)}>"

@dataclasses.dataclass(frozen=True, eq=True)
class ObserverTelem:
    ts : datetime.datetime
    email : str
    obs : str
    on : bool

    def __str__(self):
        return f"<ObserverTelem: {repr(self.obs)} {self.email} at {self.ts.strftime(PRETTY_MODIFIED_TIME_FORMAT)} [{'on' if self.on else 'off'}]>"

def parse_observers_line(line):
    match = LINE_FORMAT_REGEX.match(line)
    if match is None:
        raise RuntimeError(f"Malformed line from observers telem: " + repr(line))
    groups = match.groups()
    ts = datetime.datetime(
        year=int(groups[0]),
        month=int(groups[1]),
        day=int(groups[2]),
        hour=int(groups[3]),
        minute=int(groups[4]),
        second=int(groups[5]),
        microsecond=int(groups[6]),
        tzinfo=timezone.utc,
    )
    email = groups[7]
    obs = groups[8]
    on = groups[9] == '1'
    return ObserverTelem(ts, email, obs, on)

def parse_logdump_for_observers(telem_root : pathlib.Path, telem_path : pathlib.Path, ignore_data_integrity : bool = False):
    args = ['logdump', '--ext=.bintel', '-F', telem_path.as_posix()]
    p1 = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=LINE_BUFFERED,
        text=True
    )
    outiter = iter(p1.stdout.readline, '')
    # skip line 1: 0
    try:
        next(outiter)
    except StopIteration:
        return
    # skip line 2: path
    try:
        next(outiter)
    except StopIteration:
        return
    last_telem = None
    for line in outiter:
        line = line.strip()
        if not line:
            continue
        try:
            this_telem = parse_observers_line(line)
        except Exception as e:
            log.exception("Unparseable line running command " + " ".join(args))
        if last_telem is None:
            last_telem = this_telem
        # convert to "edge-triggered" events only by yielding
        # a line only when something is changed
        if (
            this_telem.email != last_telem.email or
            this_telem.obs != last_telem.obs or
            this_telem.on != last_telem.on
        ):
            last_telem = this_telem
            yield this_telem
        returncode = p1.poll()
        if (returncode is not None and returncode != 0) and not ignore_data_integrity:
            raise RuntimeError(f"Got exit code {returncode} from command " + " ".join(args))

@dataclasses.dataclass(frozen=True, eq=True)
class ObservationSpan:
    email : str
    title : str
    begin : datetime.datetime
    end : typing.Optional[datetime.datetime]

    def __str__(self, ):
        endpart = f" to {self.end.strftime(PRETTY_MODIFIED_TIME_FORMAT)}" if self.end is not None else ""
        return f"<ObservationSpan: {self.email} '{self.title}' from {self.begin.strftime(PRETTY_MODIFIED_TIME_FORMAT)}{endpart}>"

def xfilename_to_utc_timestamp(filename):
    _, filename = os.path.split(filename)
    name, ext = os.path.splitext(filename)
    chopped_ts_str = name.rsplit('_', 1)[1]
    chopped_ts_str = chopped_ts_str[:-3]  # nanoseconds are too precise for Python's native datetimes
    ts = datetime.datetime.strptime(chopped_ts_str, MODIFIED_TIME_FORMAT).replace(tzinfo=timezone.utc)
    return ts

def path_to_timestamped_file(the_path : pathlib.Path) -> TimestampedFile:
    try:
        ts = xfilename_to_utc_timestamp(the_path.name)
    except ValueError as e:
        log.exception(f"Unparseable filename: {the_path}")
        raise
    return TimestampedFile(the_path, ts)

def load_file_history(history_path: pathlib.Path) -> typing.Set[str]:
    history : typing.Set[str] = set()
    try:
        with history_path.open('r') as fh:
            for line in fh:
                history.add(line.strip())
    except FileNotFoundError:
        log.debug(f"Could not find lookyloo history at {history_path}")
    return history

def append_files_to_history(history_path, archive_paths, dry_run):
    if not len(archive_paths):
        return
    archive_path_lines = [f"{p.name}\n" for p in archive_paths]
    if not dry_run:
        log.debug(f"Appending to history file {history_path}: {archive_path_lines}")
        with history_path.open('a') as fh:
            fh.writelines(archive_path_lines)
    else:
        log.debug(f"dry run: appending to history file {history_path}")
        for p in archive_paths:
            log.debug(f"dry run: appending {p.name}")

def find_device_names_in_folder(base_path : pathlib.Path, extension : str) -> typing.Set[str]:
    device_names = set()
    for fn in base_path.glob(f"*.{extension}"):
        name = fn.name.rsplit('_', 1)[0]
        device_names.add(name)
    return device_names

def get_matching_paths(
    base_path : pathlib.Path,
    device : str,
    extension : str,
    newer_than_dt: datetime.datetime,
    older_than_dt: datetime.datetime=None,
    grab_one_before_start=False,
) -> typing.List[TimestampedFile]:
    newer_than_dt_fn = f"{device}_{newer_than_dt.strftime(MODIFIED_TIME_FORMAT)}000.{extension}"
    if older_than_dt is not None:
        older_than_dt_fn = f"{device}_{older_than_dt.strftime(MODIFIED_TIME_FORMAT)}000.{extension}"
    else:
        older_than_dt_fn = None
    all_matching_files = list(sorted(base_path.glob(f'{device}_*.{extension}')))
    n_files = len(all_matching_files)
    filtered_files = []
    log.debug(f"Interval endpoint filenames: {newer_than_dt_fn=} {older_than_dt_fn=}")
    for idx, the_path in enumerate(all_matching_files):
        # it's possible for a file to start before `newer_than_dt`
        # but record an observation starting after `newer_than_dt`, so
        # we grab the last file *before* the first that was started after
        # our `newer_than_dt` too
        if (
            the_path.name > newer_than_dt_fn or
            (idx + 1 < n_files and all_matching_files[idx+1].name > newer_than_dt_fn) or
            (idx == n_files - 1)
        ):
            if not grab_one_before_start and the_path.name < newer_than_dt_fn:
                # can't find any in-range entries from files that were opened before this
                # span started (assuming files start writing after spans begin, grab_one_before_start==False) so skip
                continue
            if older_than_dt is not None and the_path.name > older_than_dt_fn:
                # can't find in-range entries from files opened after `older_than_dt`
                # so skip
                continue
            filtered_files.append(path_to_timestamped_file(the_path))
    return filtered_files


def get_observation_telems(data_roots: typing.List[pathlib.Path], start_dt : datetime.datetime, end_dt : datetime.datetime, ignore_data_integrity : bool):
    events = []
    observers_data_root = None
    for data_root in data_roots:
        obs_dev_path = data_root / 'telem'
        log.debug(f"Scanning {obs_dev_path} for {OBSERVERS_DEVICE} telems")
        if len(list(obs_dev_path.glob(f'{OBSERVERS_DEVICE}_*.bintel'))):
            observers_data_root = obs_dev_path
            break
    if observers_data_root is None:
        raise RuntimeError(f"No {OBSERVERS_DEVICE} device telemetry in any of {[(x / 'telem').as_posix() for x in data_roots]}")
    for telem_path in get_matching_paths(observers_data_root, OBSERVERS_DEVICE, 'bintel', start_dt, end_dt, grab_one_before_start=True):
        events.extend(parse_logdump_for_observers(observers_data_root, telem_path.path, ignore_data_integrity))
    return events

def transform_telems_to_spans(events : typing.List[ObserverTelem], start_dt : datetime.datetime, end_dt : typing.Optional[datetime.datetime]=None):
    spans = []
    current_observer_email : str = None
    current_observation : str = None
    current_observation_start : datetime.datetime = None

    def _add_span(email, title, begin, end):
        if end is not None and end_dt is not None and end > end_dt:
            return
        if begin < start_dt:
            return
        spans.append(ObservationSpan(
            email,
            title,
            begin,
            end
        ))

    for event in events:
        if event.on:
            if current_observation_start is not None:
                # something other than 'on' must have changed, or else
                # edge-triggering must be borked
                # so we end this current span before starting the next one, guessing the timestamp
                _add_span(
                    current_observer_email,
                    current_observation,
                    current_observation_start,
                    event.ts
                )
            current_observer_email = event.email
            current_observation = event.obs
            current_observation_start = event.ts
            # log.debug(f"Began span {current_observer_email} {current_observation} {event.ts}")
        elif not event.on and current_observation is not None:
            _add_span(
                current_observer_email,
                current_observation,
                current_observation_start,
                event.ts
            )
            current_observation = current_observation_start = current_observer_email = None

    # new starting point for next iteration, ignore anything that we already processed
    if len(spans):
        start_dt = spans[-1].end

    if current_observation_start is not None:
        _add_span(
            current_observer_email,
            current_observation,
            current_observation_start,
            None
        )
    return spans, start_dt

def get_new_observation_spans(
        data_roots: typing.List[pathlib.Path],
        existing_observation_spans : typing.Set[ObservationSpan],
        start_dt : datetime.datetime,
        end_dt : typing.Optional[datetime.datetime]=None,
        ignore_data_integrity : bool=False,
) -> tuple[set[ObservationSpan, datetime.datetime]]:
    events = get_observation_telems(data_roots, start_dt, end_dt, ignore_data_integrity)
    spans, start_dt = transform_telems_to_spans(events, start_dt, end_dt)
    if len(spans):
        new_observation_spans = set(spans) - existing_observation_spans
        return new_observation_spans, start_dt
    else:
        return set(), start_dt

def prune_outputs(scratch_dir, span, exclude_before=True, exclude_after=False):
    good_outputs = []
    sorted_matches = list(sorted(scratch_dir.glob("*")))
    for fn in sorted_matches:
        if not fn.name.endswith('.fits'):
            log.debug(f"Filtered: {fn}")
            os.remove(fn)
            continue
        ts = xfilename_to_utc_timestamp(fn)
        is_before = ts < span.begin
        is_after = span.end is not None and ts > span.end
        if (exclude_before and is_before) or (exclude_after and is_after):
            os.remove(fn)
            log.debug(f"Filtered: {fn} (\n\tts =\t{ts.isoformat()},\nspan.begin =\t{span.begin.isoformat()},\nspan.end =\t{span.end.isoformat() if span.end is not None else span.end})")
        else:
            good_outputs.append(pathlib.Path(fn))
            log.debug(f"Kept: {fn}")
    return good_outputs

def datestamp_strings_from_ts(ts : datetime.datetime):
    # if the span begins before noon Chile time on the given D, the
    # stringful timestamp is "YYYY-MM-(D-1)_D" because we observe over
    # UTC night of one day into morning of the next.

    # note that daylight saving doesn't change the day, so we just use
    # UTC-4 for CLST
    # modified: roughly local Chile time for the start of the span
    modified = ts - datetime.timedelta(hours=4)

    if modified.hour < 12:
        # if span started before noon, assume it was the previous night UTC
        start_date = (modified - datetime.timedelta(days=1)).date()
    else:
        # otherwise chile date == utc date
        start_date = modified.date()
    if start_date.month > 6:
        semester = 'B'
    else:
        semester = 'A'
    end_date = start_date + datetime.timedelta(days=1)
    day_string = start_date.isoformat()
    end_part = ""
    if end_date.year != start_date.year:
        end_part += f"{end_date.year:04}-"
    if end_date.month != start_date.month:
        end_part += f"{end_date.month:02}-"
    end_part += f"{end_date.day:02}"
    day_string += "_" + end_part
    return f"{start_date.year}{semester}", day_string

def catalog_name_from_outputs(output_files):
    catobj = None
    for fn in output_files:
        log.debug(f"catalog name: {fn=}")
        if not fn.name.endswith('.fits'):
            log.debug(f"catalog name: skipped")
            continue
        with open(fn, 'rb') as fh:
            header = fits.getheader(fh)
            catobj = header.get('CATOBJ', 'invalid')
            if catobj == 'invalid' or len(catobj.replace('*', '')) == 0:
                log.debug(f"catalog name: invalid")
                catobj = None
        if catobj is not None:
            log.debug(f"catalog name: {catobj}")
            break
    if catobj is None:
        return '_no_target_'
    return catobj


def do_quicklooks_for_camera_all_roots(
    span : ObservationSpan,
    data_roots,
    device,
    omit_telemetry,
    output_dir,
    dry_run, cube_mode,
    executor : futures.ThreadPoolExecutor,
    xrif2fits_cmd,
    symlink_tree_dir: pathlib.Path=None,
    all_visited_files=None,
    ignore_history=False,
    ignore_data_integrity: bool=False,
):
    if all_visited_files is None:
        all_visited_files = set()

    found_at_least_one = False
    new_matching_files = set()

    for data_root in data_roots:
        image_path = data_root / 'rawimages' / device
        if image_path.is_dir():
            found_at_least_one = True
            new_matching_files.update(do_quicklook_for_camera(
                span,
                image_path,
                data_roots,
                device,
                omit_telemetry,
                output_dir,
                dry_run, cube_mode,
                executor,
                symlink_tree_dir=symlink_tree_dir,
                all_visited_files=all_visited_files,
                xrif2fits_cmd=xrif2fits_cmd,
                ignore_history=ignore_history,
                ignore_data_integrity=ignore_data_integrity,
            ))
    if not found_at_least_one:
        raise RuntimeError(f"Unknown device: {device} (checked {data_roots})")
    
    return new_matching_files

def do_quicklook_for_camera(
    span : ObservationSpan,
    image_path,
    data_roots,
    device,
    omit_telemetry,
    output_dir,
    dry_run, cube_mode,
    executor : futures.ThreadPoolExecutor,
    symlink_tree_dir: pathlib.Path,
    all_visited_files,
    xrif2fits_cmd,
    ignore_history,
    ignore_data_integrity,
):
    semester, night = datestamp_strings_from_ts(span.begin)
    title = f"{span.begin.strftime(FOLDER_TIMESTAMP_FORMAT)}_{span.title}"

    if not span.email:
        email = "_no_email_"
    else:
        email = span.email
    # ... / 2022B / a@b.edu / 2022-02-02_020304_label
    simple_observer_prefix = output_dir / semester / email / title

    destination = simple_observer_prefix / device
    history_path = destination / HISTORY_FILENAME
    failed_history_path = destination / FAILED_HISTORY_FILENAME
    log.debug(f"Checking {image_path} ...")
    matching_files = set(get_matching_paths(image_path, device, 'xrif', newer_than_dt=span.begin, older_than_dt=span.end, grab_one_before_start=True))
    all_visited_files = load_file_history(history_path)
    log.debug(f"Loaded previously visited files: {all_visited_files}")
    new_matching_files = set(x for x in matching_files if x.path.name not in all_visited_files)
    if len(new_matching_files) == 0 and len(all_visited_files) > 0:
        log.debug(f"Already processed all {len(matching_files)} matching files")
    if len(new_matching_files):
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        log.debug(f"Matching files:")
        for fn in sorted(new_matching_files, key=lambda x: x.timestamp):
            log.debug(f"\t{fn} (new)")
        successful_paths, failed_paths = convert_xrif(
            image_path, data_roots, new_matching_files, destination, omit_telemetry, dry_run, cube_mode,
            xrif2fits_cmd, executor, ignore_data_integrity
        )
        if not dry_run:
            log.debug("Pruning outputs that don't lie in this span")
            kept_output_files = prune_outputs(destination, span)
        else:
            log.debug("dry run: prune files outside the span")
            kept_output_files = []
        if len(kept_output_files) and symlink_tree_dir is not None and not dry_run:
            catalog_name = catalog_name_from_outputs(kept_output_files)
            # ... / a@b.edu / 2022B / Sirius / 2022-02-02_020304_label -> (../../../../2022B/a@b.edu/2022-02-02_020304_label)
            friendly_observer_semester_prefix = symlink_tree_dir / email / semester / catalog_name / night / title
            friendly_observer_semester_prefix.parent.mkdir(parents=True, exist_ok=True)
            if not friendly_observer_semester_prefix.is_symlink():
                relpath_to_simple_destination = os.path.relpath(simple_observer_prefix, start=friendly_observer_semester_prefix.parent)
                log.debug(f"Making link at {friendly_observer_semester_prefix} pointing to {relpath_to_simple_destination}")
                try:
                    friendly_observer_semester_prefix.symlink_to(relpath_to_simple_destination)
                except OSError as e:
                    log.exception(f"Could not make a symbolic link at {friendly_observer_semester_prefix} pointing to {relpath_to_simple_destination} (unsupported by destination filesystem, maybe?)")
        for archive_path in successful_paths:
            log.info(f"Converted {archive_path}")
        for archive_path in failed_paths:
            log.info(f"Failed    {archive_path}")
        if len(successful_paths) and not ignore_history:
            append_files_to_history(history_path, successful_paths, dry_run)
            log.debug(f"Wrote {len(successful_paths)} to {history_path}")
        if len(failed_paths) and not ignore_history:
            append_files_to_history(failed_history_path, failed_paths, dry_run)
            log.debug(f"{len(failed_paths)} paths failed to convert, saving to {failed_history_path}")

        log.info(f"Wrote {len(successful_paths)} output file{'s' if len(successful_paths) != 1 else ''} to {destination}")
    return new_matching_files

def launch_xrif2fits(args : list[str], telem_args : list[str], dry_run=False):
    command_line = " ".join(args + telem_args)
    if not dry_run:
        log.debug(f"Launching: {command_line}")
        try:
            proc = subprocess.Popen(
                args + telem_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            proc.wait(timeout=XRIF2FITS_TIMEOUT_SEC)
            success = proc.returncode == 0
            if not success:
                log.error(f"xrif2fits exited with nonzero exit code. Command was: {command_line}")
        except subprocess.TimeoutExpired as e:
            log.error(f"xrif2fits canceled after {XRIF2FITS_TIMEOUT_SEC} sec timeout. Command was: {command_line}")
            proc.kill()
            success = False
    else:
        success = True
        log.debug(f"dry run: {command_line}")
    if len(telem_args) != 0 and not success:
        no_telem_success, no_telem_command_line = launch_xrif2fits(args, [], dry_run=dry_run)
        if no_telem_success:
            log.error(f"Reprocessing without telemetry succeded. Original command: {command_line}, revised command: {no_telem_command_line}")
        else:
            log.error(f"Reprocessing without telemetry failed. Original command: {command_line}, revised command: {no_telem_command_line}")
    return success, command_line

def convert_xrif(
    base_dir, data_roots: typing.List[pathlib.Path],
    paths : typing.List[TimestampedFile], destination_path : pathlib.Path,
    omit_telemetry : bool, dry_run : bool, cube_mode : bool, xrif2fits_cmd: str,
    executor : futures.ThreadPoolExecutor, ignore_data_integrity : bool
):
    for data_file in paths:
        delta = datetime.datetime.utcnow().replace(tzinfo=timezone.utc) - data_file.timestamp
        if delta.total_seconds() < SLEEP_FOR_TELEMS:
            time.sleep(SLEEP_FOR_TELEMS - delta.total_seconds())
    failed_commands = []
    successful_paths, failed_paths = [], []
    pending_xrif2fits_procs = []
    task_to_ts_path_lookup = {}
    for ts_path in paths:
        args = [
            xrif2fits_cmd,
            '-d', str(base_dir),
            '-f', ts_path.path.name,
            '-D', str(destination_path),
        ]
        if cube_mode:
            args.append('--cubeMode')
        if not omit_telemetry:
            telem_paths_str = ','.join((x / 'telem').as_posix() for x in data_roots)
            log_paths_str = ','.join((x / 'logs').as_posix() for x in data_roots)

            telem_args = [
                '-t', telem_paths_str,
                '-l', log_paths_str,
            ]
        else:
            telem_args = []
        fut = executor.submit(launch_xrif2fits, args, telem_args, dry_run=dry_run)
        pending_xrif2fits_procs.append(fut)
        task_to_ts_path_lookup[fut] = ts_path

    for fut in futures.as_completed(pending_xrif2fits_procs):
        ts_path = task_to_ts_path_lookup[fut]
        success, command_line = fut.result()
        if not success:
            failed_commands.append(command_line)
            failed_paths.append(ts_path.path)
        else:
            successful_paths.append(ts_path.path)

    if len(failed_commands):
        log.debug(f"failed commands:")
        for cmd in failed_commands:
            log.debug(f"\tfailed: {cmd}")
    log.debug(f"Extracted {len(paths)} XRIF archives to FITS")
    return successful_paths, failed_paths

def decide_to_process(args, span):
    if args.title is not None:
        title_match = span.title.strip().lower() == args.title.strip().lower()
        if args.partial_match_ok:
            title_match = title_match or args.title.strip().lower() in span.title.lower()
    else:
        title_match = True

    if args.observer_email is not None:
        observer_match = args.observer_email.strip().lower() in span.email.lower()
    else:
        observer_match = True

    return title_match and observer_match

def process_span(
    span : ObservationSpan,
    output_dir : pathlib.Path, cameras : typing.List[str],
    data_roots: typing.List[pathlib.Path], omit_telemetry : bool,
    xrif2fits_cmd: str,
    all_visited_files: typing.List[TimestampedFile],
    ignore_history: bool,
    executor : futures.ThreadPoolExecutor,
    dry_run : bool,
    force_cube_or_separate : typing.Optional[object]=None,
    symlink_tree_dir: pathlib.Path=None,
    ignore_data_integrity:bool=False,
):
    log.info(f"Observation interval to process: {span}")
    for device in cameras:
        cube_mode = ALL_CAMERAS.get(device, DEFAULT_SEPARATE) is DEFAULT_CUBE
        if force_cube_or_separate is DEFAULT_CUBE:
            cube_mode = True
        elif force_cube_or_separate is DEFAULT_SEPARATE:
            cube_mode = False
        visited_files = do_quicklooks_for_camera_all_roots(
            span,
            data_roots,
            device,
            omit_telemetry or cube_mode,  # can't use telem in cube mode so always omit
            output_dir,
            dry_run,
            cube_mode,
            executor,
            xrif2fits_cmd,
            symlink_tree_dir=symlink_tree_dir,
            all_visited_files=all_visited_files,
            ignore_history=ignore_history,
            ignore_data_integrity=ignore_data_integrity,
        )
        all_visited_files = all_visited_files.union(visited_files)
        if len(visited_files) and (span.end is not None and (utcnow() - span.end).total_seconds() > 5 * 60):
            log.info(f"Exported {len(visited_files)} archive{'s' if len(visited_files) != 1 else ''} for {device}")

    log.info(f"Completed span {span}")

def _copy_task(src : str, dest : str, dry_run : bool) -> str:
    if not dry_run:
        shutil.copy(src, dest)
    else:
        log.debug(f"dry run: cp {src} {dest}")
    return dest

def stage_for_bundling(
    data_root : pathlib.Path,
    data_files : typing.List[TimestampedFile],
    bundle_root : pathlib.Path,
    dry_run : bool,
    threadpool : futures.ThreadPoolExecutor,
) -> typing.List[pathlib.Path]:
    copy_tasks = []
    for fn in data_files:
        relpath = fn.path.relative_to(data_root)
        dest = bundle_root / relpath
        copy_tasks.append(threadpool.submit(_copy_task, fn.path, dest, dry_run))
    dest_paths = [dst for dst in futures.as_completed(copy_tasks)]
    return dest_paths

def _check_in_span(args : typing.List[str], span : ObservationSpan, partial_overlap_ok : bool=True):
    '''Check whether a given archive lies in the ObservationSpan
    '''
    try:
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode('utf8').strip()
        if not len(out):
            log.debug(f"No output from {' '.join(args)}")
            return False
        _, start_isostamp, end_isostamp, _, _ = out.split(' ')
    except Exception as e:
        log.exception(f"Checking binary archive endpoints ({' '.join(args)}) failed with:\n{out} ")
        raise
    start_ts = datetime.datetime.fromisoformat(start_isostamp[:-4]).replace(tzinfo=timezone.utc)
    end_ts = datetime.datetime.fromisoformat(end_isostamp[:-4]).replace(tzinfo=timezone.utc)
    if not partial_overlap_ok:
        return (
            (start_ts >= span.begin) and
            (end_ts <= span.end)
        )
    else:
        return (
            (end_ts >= span.begin) or
            (start_ts <= span.end)
        )

def check_xrif_in_span(span : ObservationSpan, data_roots: typing.List[pathlib.Path], xrif_file : TimestampedFile):
    '''Use the --time mode of xrif2fits to check whether a given archive lies in the ObservationSpan
    '''
    args = [
        "/usr/local/bin/xrif2fits",
        "--time",
        '-d', str(xrif_file.path.parent),
        '-f', xrif_file.path.name,
    ]
    telem_paths_str = ','.join((x / 'telem').as_posix() for x in data_roots)
    log_paths_str = ','.join((x / 'logs').as_posix() for x in data_roots)
    args.extend([
        '-t', telem_paths_str,
        '-l', log_paths_str,
    ])
    return _check_in_span(args, span)


def check_log_in_span(span : ObservationSpan, log_file : TimestampedFile):
    '''Use the --time mode of logdump to check whether a given archive lies in the ObservationSpan
    '''
    args = [
        "/usr/local/bin/logdump",
        "--time",
        '-d', str(log_file.path.parent),
        '-f', log_file.path.name,
    ]
    return _check_in_span(args, span)


def check_telem_in_span(span : ObservationSpan, log_file : TimestampedFile):
    '''Use the --time mode of logdump for .bintel to check whether a given archive lies in the ObservationSpan
    '''
    args = [
        "/usr/local/bin/logdump",
        "--ext=.bintel",
        "--time",
        '-d', str(log_file.path.parent),
        '-f', log_file.path.name,
    ]
    return _check_in_span(args, span)

def create_bundle_from_span(
    span : ObservationSpan,
    output_dir : pathlib.Path,
    data_roots: typing.List[pathlib.Path],
    threadpool : futures.ThreadPoolExecutor,
    dry_run : bool,
    cameras : typing.Optional[typing.List[str]] = None,
):
    log.info(f"Observation interval to bundle: {span}")
    bundle_root = output_dir / f'bundle_{format_timestamp_for_filename(span.begin)}_to_{format_timestamp_for_filename(span.end)}'
    log.debug(f"Staging to {bundle_root.as_posix()}")
    for subfolder in ['logs', 'telem', 'rawimages']:
        subfolder_path = bundle_root / subfolder
        if not dry_run:
            os.makedirs(subfolder_path, exist_ok=True)
            log.debug(f"mkdir {subfolder_path}")
        else:
            log.debug(f"dry run: mkdir {subfolder_path}")

    bundle_contents = []
    for data_root in data_roots:
        # collect logs
        logs_root = data_root / 'logs'
        for devname in find_device_names_in_folder(logs_root, 'binlog'):
            log_files = get_matching_paths(
                logs_root,
                device=devname,
                extension='binlog',
                newer_than_dt=span.begin,
                older_than_dt=span.end,
                grab_one_before_start=True,
            )
            if not len(log_files):
                continue
            elif not check_log_in_span(span, log_files[0]):
                # drop first file if it was grabbed by optimistic matching for pre-interval-start frames
                log_files = log_files[1:]
            bundle_contents.extend(stage_for_bundling(
                data_root,
                log_files,
                bundle_root,
                dry_run,
                threadpool,
            ))
        # collect telems
        telem_root = data_root / 'telem'
        for devname in find_device_names_in_folder(telem_root, 'bintel'):
            telem_files = get_matching_paths(
                telem_root,
                device=devname,
                extension='bintel',
                newer_than_dt=span.begin,
                older_than_dt=span.end,
                grab_one_before_start=True,
            )
            if not len(telem_files):
                continue
            elif not check_telem_in_span(span, telem_files[0]):
                # drop first file if it was grabbed by optimistic matching for pre-interval-start frames
                telem_files = telem_files[1:]
            bundle_contents.extend(stage_for_bundling(
                data_root,
                telem_files,
                bundle_root,
                dry_run,
                threadpool,
            ))
        # for every camera:
        if cameras is not None:
            images_dirs = list(filter(lambda x: x.is_dir(), [data_root / 'rawimages' / camname for camname in cameras]))
        else:
            images_dirs = list(filter(lambda x: x.is_dir(), (data_root / 'rawimages').glob('*')))
        log.debug(f"{data_root=} {images_dirs=}")
        for imgdir in images_dirs:
            log.debug(f"{imgdir=}")
            # collect image archives
            cam_files = get_matching_paths(
                imgdir,
                device=imgdir.name,
                extension='xrif',
                newer_than_dt=span.begin,
                older_than_dt=span.end,
                grab_one_before_start=True,
            )
            if not len(cam_files):
                continue
            elif not check_xrif_in_span(span, data_roots, cam_files[0]):
                # drop first file if it was grabbed by optimistic matching for pre-interval-start frames
                cam_files = cam_files[1:]
            # only make dir if there's a nonzero number of camera archives
            images_dest = bundle_root / 'rawimages' / imgdir.name
            if not dry_run:
                os.makedirs(images_dest, exist_ok=True)
                log.debug(f"mkdir {images_dest}")
            else:
                log.debug(f"dry run: mkdir {images_dest}")
            bundle_contents.extend(stage_for_bundling(
                data_root,
                cam_files,
                bundle_root,
                dry_run,
                threadpool,
            ))
    return bundle_root
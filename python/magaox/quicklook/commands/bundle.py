import time
import os
import typing
from concurrent import futures
import logging
import datetime
from datetime import timezone
import pathlib
import argparse
from ...constants import HISTORY_FILENAME, ALL_CAMERAS, LOOKYLOO_DATA_ROOTS, QUICKLOOK_PATH, DEFAULT_CUBE, DEFAULT_SEPARATE, CHECK_INTERVAL_SEC, LOG_PATH
from ...utils import parse_iso_datetime, format_timestamp_for_filename, utcnow, get_search_start_end_timestamps
from ..core import (
    TimestampedFile,
    ObservationSpan,
    load_file_history,
    do_quicklook_for_camera,
    get_new_observation_spans,
    process_span,
    decide_to_process,
    create_bundle_from_span,
)

import upath


import xconf
from ._base import BaseQuicklookCommand

from ... import constants

log = logging.getLogger(__name__)

@xconf.config
class Bundle(BaseQuicklookCommand):
    cameras : list[str] = xconf.field(default_factory=lambda: list(constants.ALL_CAMERAS.keys()), help="Camera names (i.e. rawimages subfolder names)")
    output_dir : upath.UPath = xconf.field(default=upath.UPath('.'), help="Path or URI to destination")

    def main(self):
        if not self.output_path.is_dir():
            self.output_path.mkdir(parents=True, exist_ok=True)
        timestamp_str = format_timestamp_for_filename(utcnow())
        log_file_path = f"./lookyloo_bundle_{timestamp_str}.log" if args.verbose or args.dry_run else None
        log_format = '%(filename)s:%(lineno)d: [%(levelname)s] %(message)s'
        logging.basicConfig(
            level='DEBUG' if args.verbose or args.dry_run else 'INFO',
            filename=log_file_path,
            format=log_format
        )
        # Specifying a filename results in no console output, so add it back
        if args.verbose or args.dry_run:
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logging.getLogger('').addHandler(console)
            formatter = logging.Formatter(log_format)
            console.setFormatter(formatter)
            log.debug(f"Logging to {log_file_path}")

        cameras = args.camera
        if args.data_root:
            data_roots = [pathlib.Path(x) for x in args.data_root]
        else:
            data_roots = [pathlib.Path(x) for x in LOOKYLOO_DATA_ROOTS.split(':')]
        output_dir = pathlib.Path(args.output_dir)
        start_dt, end_dt = get_search_start_end_timestamps(args.semester, args.utc_start, args.utc_end)
        new_observation_spans, _ = get_new_observation_spans(data_roots, set(), start_dt, end_dt, ignore_data_integrity=args.ignore_data_integrity)

        with futures.ThreadPoolExecutor(max_workers=args.parallel_jobs) as threadpool:
            for span in new_observation_spans:
                if span.end is None:
                    log.debug(f"Skipping {span} because it is an open interval")
                    continue
                if decide_to_process(args, span):
                    log.info(f"Observation interval to process: {span}")
                    create_bundle_from_span(
                        span,
                        output_dir,
                        data_roots,
                        threadpool,
                        args.dry_run,
                        cameras,
                    )

def main():
    now = datetime.datetime.now()
    this_year = now.year
    this_semester = str(this_year) + ("B" if now.month > 6 else "A")
    parser = argparse.ArgumentParser(description="Bundle observations for upload")
    parser.add_argument('-r', '--dry-run', help="Commands to run are printed in debug output (implies --verbose)", action='store_true')
    parser.add_argument('-v', '--verbose', help="Turn on debug output", action='store_true')
    parser.add_argument('-t', '--title', help="Title of observation to collect", action='store')
    parser.add_argument('-e', '--observer-email', help="Skip observations that are not by this observer (matches substrings, case-independent)", action='store')
    parser.add_argument('-p', '--partial-match-ok', help="A partial match (title provided is found anywhere in recorded title) is processed", action='store_true')
    parser.add_argument('-s', '--semester', help=f"Semester to search in, default: {this_semester}", default=this_semester)
    parser.add_argument('--utc-start', help=f"ISO UTC datetime stamp of earliest observation start time to process (supersedes --semester)", type=parse_iso_datetime)
    parser.add_argument('--utc-end', help=f"ISO UTC datetime stamp of latest observation end time to process (ignored in daemon mode)", type=parse_iso_datetime)
    parser.add_argument('-c', '--camera', help=f"Camera name (i.e. rawimages subfolder name), repeat to specify multiple names. (default: all XRIF-writing cameras found)", action='append')
    parser.add_argument('-X', '--data-root', help=f"Search directory for telem and rawimages subdirectories, repeat to specify multiple roots. (default: {LOOKYLOO_DATA_ROOTS.split(':')})", action='append')
    parser.add_argument('-D', '--output-dir', help=f"output directory, defaults to current dir", action='store', default=os.getcwd())
    parser.add_argument('-j', '--parallel-jobs', default=8, help="Max number of parallel xrif2fits processes to launch (default: 8; if the number of archives in an interval is smaller than this, fewer processes will be launched)")
    parser.add_argument('--ignore-data-integrity', help="[DEBUG USE ONLY]", action='store_true')
    args = parser.parse_args()
    output_path = pathlib.Path(args.output_dir)
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    timestamp_str = format_timestamp_for_filename(utcnow())
    log_file_path = f"./lookyloo_bundle_{timestamp_str}.log" if args.verbose or args.dry_run else None
    log_format = '%(filename)s:%(lineno)d: [%(levelname)s] %(message)s'
    logging.basicConfig(
        level='DEBUG' if args.verbose or args.dry_run else 'INFO',
        filename=log_file_path,
        format=log_format
    )
    # Specifying a filename results in no console output, so add it back
    if args.verbose or args.dry_run:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        formatter = logging.Formatter(log_format)
        console.setFormatter(formatter)
        log.debug(f"Logging to {log_file_path}")

    cameras = args.camera
    if args.data_root:
        data_roots = [pathlib.Path(x) for x in args.data_root]
    else:
        data_roots = [pathlib.Path(x) for x in LOOKYLOO_DATA_ROOTS.split(':')]
    output_dir = pathlib.Path(args.output_dir)
    start_dt, end_dt = get_search_start_end_timestamps(args.semester, args.utc_start, args.utc_end)
    new_observation_spans, _ = get_new_observation_spans(data_roots, set(), start_dt, end_dt, ignore_data_integrity=args.ignore_data_integrity)

    with futures.ThreadPoolExecutor(max_workers=args.parallel_jobs) as threadpool:
        for span in new_observation_spans:
            if span.end is None:
                log.debug(f"Skipping {span} because it is an open interval")
                continue
            if decide_to_process(args, span):
                log.info(f"Observation interval to process: {span}")
                create_bundle_from_span(
                    span,
                    output_dir,
                    data_roots,
                    threadpool,
                    args.dry_run,
                    cameras,
                )
import enum
import time
import os
import typing
from concurrent import futures
import logging
import datetime
from datetime import timezone
import pathlib
import argparse

import xconf
import zarr

from ... import constants
from ...constants import (
    HISTORY_FILENAME,
    ALL_CAMERAS,
    LOOKYLOO_DATA_ROOTS,
    QUICKLOOK_PATH,
    DEFAULT_CUBE,
    DEFAULT_SEPARATE,
    CHECK_INTERVAL_SEC,
    LOG_PATH,
)
from ... import utils
from ...utils import parse_iso_datetime, format_timestamp_for_filename, utcnow
from ..utils import get_search_start_end_timestamps
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
from ..pack import pack_one_obs, StreamConfig, DEFAULT_STREAMS
from ._base import BaseQuicklookCommand


log = logging.getLogger(__name__)


class ZarrMode(enum.Enum):
    WRITE_NO_OVERWRITE = "w-"
    WRITE_WITH_OVERWRITE = "w"
    READ_ONLY = "r"
    READ_WRITE_CREATE = "r+"
    READ_WRITE = "a"


@xconf.config
class Pack(BaseQuicklookCommand):
    name: str = xconf.field(default="pack.zarr", help="Name of Zarr archive to create")
    destination: pathlib.Path = xconf.field(
        default=pathlib.Path(".").absolute(), help="xPath or URI to destination"
    )
    parallel_jobs: int = xconf.field(
        default=10, help="How many export jobs to start in parallel"
    )
    streams: list[StreamConfig] = xconf.field(
        default_factory=lambda: [StreamConfig(name=strm) for strm in DEFAULT_STREAMS]
    )
    zarr_mode: ZarrMode = xconf.field(default=ZarrMode.WRITE_NO_OVERWRITE)

    def main(self):
        if not self.destination.is_dir():
            self.destination.mkdir(parents=True, exist_ok=True)
        log.debug(f"Starting the packer with {self.destination / self.name}")
        start_time_sec = time.time()

        dbconn = self.database.connect()

        start_dt, end_dt = self.get_time_range()
        new_observation_spans, _ = get_new_observation_spans(
            start_dt,
            end_dt,
            email=self.email,
            title=self.title,
            exact_title=self.exact_title,
        )
        if len(new_observation_spans):
            log.info("Found these observation spans to pack:")
            for span in new_observation_spans:
                log.info(f"\t{span}")
        else:
            log.error("No matching observation spans to pack")
            return

        # Don't open the destination archive unless we found something to pack
        # (avoid creating empty archives)
        root = zarr.open(self.destination / self.name, mode=self.zarr_mode.value)

        total_files, orig_total_bytes, final_total_bytes = 0, 0, 0

        with futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as threadpool:
            for span in new_observation_spans:
                if span.end is None:
                    log.debug(f"Skipping {span} because it is an open interval")
                    continue
                title_key = utils.make_filename_safe(span.title)
                span_key = utils.format_timestamp_for_filename(span.begin)
                if title_key in root:
                    pack_target_group = root[title_key].require_group(span_key)
                else:
                    pack_target_group = root.require_group(f"{title_key}/{span_key}")

                log.info(f"Observation interval to process: {span}")
                paths_packed, orig_bytes, final_bytes = pack_one_obs(
                    span,
                    self.streams,
                    pack_target_group,
                    dbconn,
                    self.path_rewrites,
                    threadpool,
                    self.temp_root,
                )
                orig_total_bytes += orig_bytes
                final_total_bytes += final_bytes
                n_files = len(paths_packed)
                total_files += n_files
                if final_bytes > 0:
                    log.info(
                        f"Packed {orig_bytes/1024/1024:1.1f} MiB from {n_files} files into "
                        f"{final_bytes/1024/1024:1.1f} MiB in {pack_target_group} ({orig_bytes / final_bytes:1.2f})"
                    )
        if final_total_bytes > 0:
            duration_sec = time.time() - start_time_sec
            log.info(
                f"Packed {orig_total_bytes/1024/1024:1.1f} MiB from {total_files} files into "
                f"{final_total_bytes/1024/1024:1.1f} MiB ({orig_total_bytes / final_total_bytes:1.2f})"
            )
            log.info(f"Spent {duration_sec / 60:1.2f} min total packing")
        else:
            log.info("No data were found to pack")

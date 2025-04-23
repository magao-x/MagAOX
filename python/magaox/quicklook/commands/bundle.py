from concurrent import futures
import os
import pathlib
import logging
from ...utils import format_timestamp_for_filename
from ..core import (
    get_new_observation_spans,
    create_bundle_from_span,
)

import xconf
from ._base import BaseQuicklookCommand


log = logging.getLogger(__name__)

@xconf.config
class Bundle(BaseQuicklookCommand):
    destination : pathlib.Path = xconf.field(default=pathlib.Path('.'), help="Path or URI to destination")
    parallel_jobs : int = xconf.field(default=10, help="How many export jobs to start in parallel")
    dry_run : bool = xconf.field(default=False, help="Whether to perform a dry run or actually execute the necessary commands")

    def main(self):
        if not self.destination.is_dir():
            self.destination.mkdir(parents=True, exist_ok=True)

        start_dt, end_dt = self.get_time_range()
        new_observation_spans, _ = get_new_observation_spans(start_dt, end_dt, email=self.email, title=self.title)

        with futures.ThreadPoolExecutor(max_workers=self.parallel_jobs) as threadpool:
            for span in new_observation_spans:
                if span.end is None:
                    log.debug(f"Skipping {span} because it is an open interval")
                    continue
                dest_dir = self.destination / f'{format_timestamp_for_filename(span.begin)}_{span.title}'
                os.makedirs(dest_dir)
                log.info(f"Observation interval to process: {span}")
                dest_paths = create_bundle_from_span(
                    span,
                    dest_dir,
                    self.path_rewrites,
                    threadpool,
                    self.dry_run,
                    self.common_path_prefix,
                )
                log.debug(f"Bundled {len(dest_paths)} files to {dest_dir}")

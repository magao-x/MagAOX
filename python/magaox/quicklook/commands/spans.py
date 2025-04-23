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
class Spans(BaseQuicklookCommand):
    destination : pathlib.Path = xconf.field(default=pathlib.Path('.'), help="Path or URI to destination")

    def main(self):
        start_dt, end_dt = self.get_time_range()
        new_observation_spans, _ = get_new_observation_spans(start_dt, end_dt, email=self.email, title=self.title)
        for span in new_observation_spans:
            print(f"{span.begin.isoformat()}\t{span.end.isoformat()}\t{repr(span.email)}\t{repr(span.title)}")

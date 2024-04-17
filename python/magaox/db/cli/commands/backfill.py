import xconf
import pathlib
from datetime import timezone
import datetime
import logging
import os
import os.path
import psycopg2
import socket
import typing
from tqdm import tqdm

from magaox.db import FileOrigin, ingest
from magaox.db.config import DEFAULT_DATA_DIRS

import xconf
from ._base import BaseDbCommand

log = logging.getLogger(__name__)

@xconf.config
class Backfill(BaseDbCommand):
    '''Process ``.bintel`` files found in data folders that don't already have ingest records
    and populate the ``telem`` table
    '''
    telem_dir : str = xconf.field(default='/opt/MagAOX/telem', help="Directory where .bintel files are written")
    devices : typing.Union[list[str], None] = xconf.field(default=None, help="List of devices to consider for backfilling, leave empty for all (based on presence of .bintels, not proclist entry)")

    def _identify_devices(self):
        # scan telem_dir and build a list of device names based on bintel files present
        return devices

    def main(self):
        devices = self.devices if self.devices is not None else self._identify_devices()
        with self.database.cursor() as cur:
            ingest.backfill_telemetry(cur, self.hostname, devices)
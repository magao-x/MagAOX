import xconf
import pathlib
from datetime import timezone
import datetime
import logging
import os
import os.path
import psycopg
import socket
from tqdm import tqdm

from magaox.db import FileOrigin, ingest
from magaox.db.config import DEFAULT_DATA_DIRS

import xconf
from ._base import BaseDbCommand

log = logging.getLogger(__name__)

@xconf.config
class Inventory(BaseDbCommand):
    '''Find files that aren't yet inventoried and create records for
    them in the file_origins table
    '''
    data_dirs : list[str] = xconf.field(default_factory=lambda: DEFAULT_DATA_DIRS)

    def main(self):
        with self.database.cursor() as cur:
            ingest.update_file_inventory(cur, self.hostname, self.data_dirs)
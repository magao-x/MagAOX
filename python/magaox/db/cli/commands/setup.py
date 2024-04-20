import pathlib
import logging

import psycopg

import xconf
from ._base import BaseDbCommand

log = logging.getLogger(__name__)

SETUP_SQL_PATH = pathlib.Path(__file__).parent / '..' / '..' / 'sql'

SETUP_SQL_FILES = ['setup_tables_and_indices.sql', 'setup_views.sql']

@xconf.config
class Setup(BaseDbCommand):
    '''Create tables and indices that are not already present in the configured database
    '''

    def initialize(self, conn : psycopg.Connection):
        c = conn.cursor()
        for fn in SETUP_SQL_FILES:
            sql_fpath = SETUP_SQL_PATH / fn
            log.debug(f"Loading SQL from {sql_fpath}")
            init_sql = sql_fpath.read_text()
            if not len(init_sql.strip()):
                log.debug(f"Skipping {fn} because it's empty")
                continue
            log.debug("Running SQL:\n\n" + init_sql + "\n\n")
            c.execute(init_sql)
            c.execute("COMMIT")
        return

    def main(self):
        conn = self.database.connect()
        self.initialize(conn)
        log.info("Success!")
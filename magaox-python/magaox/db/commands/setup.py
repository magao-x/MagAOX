import pathlib
import logging

import psycopg

import xconf
from ._base import BaseDbCommand
from .. import fbs_to_sql
from ... import constants

log = logging.getLogger(__name__)

SETUP_SQL_PATH = pathlib.Path(__file__).parent / '..' / 'sql'

SETUP_SQL_FILES = ['setup_tables_and_indices.sql', 'setup_views.sql']

@xconf.config
class Setup(BaseDbCommand):
    '''Create tables and indices that are not already present in the configured database
    '''
    schema_folder : pathlib.Path = xconf.field(default=(constants.DEFAULT_PREFIX / 'source' / 'MagAOX' / 'libMagAOX' / 'logger' / 'types' / 'schemas'))
    generate_only : bool = xconf.field(default=False, help="Whether to run the SQL generation from the flatlogs / flatbuffers schemata and output it")

    def generate_tables_sql(self):
        sql_text = ""
        for fn in self.schema_folder.glob('*.fbs'):
            schema_text = open(fn).read()
            one_sql_text = fbs_to_sql.fbs_to_sql(fn.stem, schema_text)
            sql_text += one_sql_text + '\n'
        return sql_text

    def initialize(self, conn : psycopg.Connection, telem_table_creation_sql : str):
        c = conn.cursor()
        with conn.transaction():
            for sql_fpath in sorted(SETUP_SQL_PATH.glob('*.sql')):
                log.debug(f"Loading SQL from {sql_fpath}")
                init_sql = sql_fpath.read_text()
                if not len(init_sql.strip()):
                    log.debug(f"Skipping {sql_fpath} because it's empty")
                    continue
                log.debug("Running SQL:\n\n" + init_sql + "\n\n")
                c.execute(init_sql)
            log.debug("Running telem type-specific table creation SQL:\n\n" + telem_table_creation_sql + "\n\n")
            c.execute(telem_table_creation_sql)

    def main(self):
        telem_table_creation_sql = self.generate_tables_sql()
        if self.generate_only:
            print(telem_table_creation_sql)
            return
        for conn_name in self.databases:
            conn = self.databases[conn_name].connect()
            self.initialize(conn, telem_table_creation_sql)
            log.info(f"Initialized {conn}")
        log.info("Success!")

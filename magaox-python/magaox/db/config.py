import logging
import os
import pathlib
import re
import socket

import psycopg
import psycopg.rows
import xconf

from magaox.indi.device import BaseConfig as IndiDeviceBaseConfig
from ..constants import DEFAULT_DATA_DIRS

log = logging.getLogger(__name__)

__all__ = [
    'DEFAULT_DATA_DIRS',
    'DbConfig',
    'BaseConfig',
    'BaseDbDeviceConfig',
]

SETUP_USERS_SQL_PATH = pathlib.Path(__file__).parent / 'sql' / 'setup_users.sql'


@xconf.config
class DbConfig:
    host : str = xconf.field(default='localhost', help='Hostname on which PostgreSQL is listening for connections')
    user : str = xconf.field(default='xtelem', help='Username with access to PostgreSQL database over TCP')
    port : int = xconf.field(default=5432, help='TCP port to connect to PostgreSQL on')
    database : str = xconf.field(default='xtelem', help='Name of PostgreSQL database')
    password_file : str = xconf.field(default='/opt/MagAOX/secrets/xtelemdb_password', help="File containing the password for the given user (newlines are stripped). If $XTELEMDB_PASSWORD is set in the environment, it will take precedence.")
    statement_timeout_sec : float = xconf.field(default=60.0, help="Server-side enforced statement timeout")
    lock_timeout_sec : float = xconf.field(default=60.0, help="Server-side enforced lock acquisition timeout")
    idle_in_transaction_timeout_sec : float = xconf.field(default=60.0, help="Server-side enforced idle (abandoned) transaction timeout")

    def connect(self) -> psycopg.Connection:
        password = os.environ.get('XTELEMDB_PASSWORD', None)
        if password is None and os.path.exists(self.password_file):
            try:
                password = open(self.password_file, 'r').read().strip()
            except Exception:
                log.error(f"Tried to get password from {self.password_file}")

        try:
            conn = psycopg.connect(
                dbname=self.database,
                host=self.host,
                user=self.user,
                password=password,
                row_factory=psycopg.rows.dict_row,
            )
        except Exception as e:
            log.exception("Unable to connect to database.")
            log.error(f"May need password to connect to host={self.host} database={self.database} user={self.user}, "
                      f"set $XTELEMDB_PASSWORD in the environment or write in {self.password_file}")
            log.error(f"""
Also, ensure:
1. PostgreSQL is running on {self.host}:{self.port} (`systemctl status postgresql` on {self.host})
2. The database {repr(self.database)} exists
3. The appropriate user accounts have been created
4. Login over TCP is enabled, and the firewall has been configured to allow this (Hint: Use the hostname `aoc` to enforce routing over instrument LAN)

See /opt/MagAOX/source/MagAOX/setup/steps/configure_postgresql.sh for details.
""")
            raise
        return conn

    def cursor(self) -> psycopg.Cursor:
        return self.connect().cursor()



@xconf.config
class IgnorePatternsConfig:
    files : list[str] = xconf.field(default_factory=lambda: [r'.*\.DS_Store', r'.+\.swp', r'.+~'], help="Regular expression patterns to match against full file paths")
    directories : list[str] = xconf.field(default_factory=lambda: [r'.*\.git.*'], help="Regular expression patterns to match against full directory paths")

@xconf.config
class BaseConfig:
    '''Base class for telemdb commands providing a `db` config item
    '''
    databases : dict[str,DbConfig] = xconf.field(default_factory=lambda: {'local': DbConfig()}, help="PostgreSQL database connections")
    hostname : str = xconf.field(default=socket.gethostname(), help="Hostname to identify this computer when running inventory or watch_files")
    data_dirs : list[pathlib.Path] = xconf.field(default_factory=lambda: DEFAULT_DATA_DIRS.copy(), help="Inventoried/archived data directories")
    ignore_patterns : IgnorePatternsConfig = xconf.field(default_factory=IgnorePatternsConfig, help="Patterns for files and directories to ignore in the inventory")

@xconf.config
class BaseDbDeviceConfig(BaseConfig, IndiDeviceBaseConfig):
    '''Base config for devices accessing the telem db
    '''
    pass

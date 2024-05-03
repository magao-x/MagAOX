import os
import typing
import xconf
import logging
import pathlib
import psycopg
import psycopg.rows
import socket

from magaox.indi.device import BaseConfig as IndiDeviceBaseConfig

log = logging.getLogger(__name__)

__all__ = [
    'DEFAULT_DATA_DIRS',
    'DbConfig',
    'BaseConfig',
    'BaseDeviceConfig',
]

DEFAULT_DATA_DIRS = [
    '/opt/MagAOX/logs',
    '/opt/MagAOX/rawimages',
    '/opt/MagAOX/telem',
    '/opt/MagAOX/cacao',
]

SETUP_USERS_SQL_PATH = pathlib.Path(__file__).parent / 'sql' / 'setup_users.sql'

@xconf.config
class DbConfig:
    host : str = xconf.field(default='localhost', help='Hostname on which PostgreSQL is listening for connections')
    user : str = xconf.field(default='xtelem', help='Username with access to PostgreSQL database over TCP')
    port : int = xconf.field(default=5432, help='TCP port to connect to PostgreSQL on')
    database : int = xconf.field(default='xtelem', help='Name of PostgreSQL database')
    password_file : str = xconf.field(default='/opt/MagAOX/secrets/xtelemdb_password', help="File containing the password for the given user (newlines are stripped). If $XTELEMDB_PASSWORD is set in the environment, it will take precedence.")

    def connect(self) -> psycopg.Connection:
        password = os.environ.get('XTELEMDB_PASSWORD', None)
        if password is None and os.path.exists(self.password_file):
            try:
                password = open(self.password_file, 'r').read().strip()
            except Exception:
                log.error(f"Tried to get password from {self.password_file}")
                raise

        if password is None:
            raise RuntimeError(f"Need password to connect to host={self.host} database={self.database} user={self.user}, "
                               f"set $XTELEMDB_PASSWORD in the environment or write in {self.password_file}")

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
            log.error(f"""
Ensure:
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
class BaseConfig:
    '''Base class for telemdb commands providing a `db` config item
    '''
    database : DbConfig = xconf.field(default=DbConfig(), help="PostgreSQL database connection")
    hostname : str = xconf.field(default=socket.gethostname(), help="Hostname to identify this computer when running inventory or watch_files")
    data_dirs : list[str] = xconf.field(default_factory=lambda: DEFAULT_DATA_DIRS.copy(), help="Inventoried/archived data directories")

@xconf.config
class BaseDeviceConfig(BaseConfig, IndiDeviceBaseConfig):
    '''Base config for devices accessing the telem db
    '''
    pass

import os
import typing
import xconf
import logging
import pathlib
import psycopg2
import socket

from magaox.indi.device import BaseConfig as IndiDeviceBaseConfig

log = logging.getLogger(__name__)

DEFAULT_DATA_DIRS = [
    '/opt/MagAOX/logs',
    '/opt/MagAOX/rawimages',
    '/opt/MagAOX/telem',
    '/opt/MagAOX/cacao',
]

SETUP_USERS_SQL_PATH = pathlib.Path(__file__).parent / 'sql' / 'setup_users.sql'

@xconf.config
class DbConfig:
    host : str = xconf.field(default='/var/run/postgresql', help='Hostname on which PostgreSQL is listening for connections')
    user : str = xconf.field(default='xsup', help='Username with access to PostgreSQL database over TCP')
    port : int = xconf.field(default=5432, help='TCP port to connect to PostgreSQL on')
    database : int = xconf.field(default='xtelem', help='Name of PostgreSQL database')

    def connect(self) -> psycopg2.extensions.connection:
        password = os.environ.get('XTELEMDB_PASSWORD', None)
        try:
            conn = psycopg2.connect(
                database=self.database,
                host=self.host,
                user=self.user,
                password=password,
            )
        except Exception as e:
            log.error("Unable to connect to database.")
            if password is None and self.host[0] != '/':
                log.error(f"Need password to connect to host={self.host} database={self.database} user={self.user}, set $XTELEMDB_PASSWORD in the environment")
            log.error(f"""
Ensure:
1. PostgreSQL is running on {self.host}:{self.port} (`systemctl status postgresql` on {self.host})
2. The database {repr(self.database)} exists (to create, on host {self.host}, do `sudo -u postgres psql -U postgres -c "CREATE DATABASE {self.database}"`)
3. The appropriate user accounts have been created (on host {self.host}, do `sudo -u postgres psql < {SETUP_USERS_SQL_PATH}`)
4. Login over TCP is enabled, and the firewall has been configured to allow this (Hint: Use the hostname `aoc` to enforce routing over instrument LAN)
""")
            raise
        return conn

    def cursor(self) -> psycopg2.extensions.cursor:
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

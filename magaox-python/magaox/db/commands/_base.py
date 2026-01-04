import socket

import psycopg
import xconf

import magaox.db.config as dbconfig

@xconf.config
class BaseDbCommand(dbconfig.BaseConfig, xconf.Command):
    def main(self):
        raise NotImplementedError("Command subclasses must implement main()")

import socket

import psycopg
import xconf

import magaox.db.config as config

@xconf.config
class BaseDbCommand(config.BaseConfig, xconf.Command):
    def main(self):
        raise NotImplementedError("Command subclasses must implement main()")

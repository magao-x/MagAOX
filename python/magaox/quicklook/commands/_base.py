import datetime
import socket
import typing
import psycopg
import xconf

import magaox.db.config as dbconfig

from ... import constants

@xconf.config
class BaseQuicklookCommand(dbconfig.BaseConfig, xconf.Command):
    dry_run : bool = xconf.field(default=False, help="Whether to perform a dry run or actually execute the necessary commands")
    title : typing.Optional[str] = xconf.field(default=None, help="All or part of the observation name to process")
    email : typing.Optional[str] = xconf.field(default=None, help="Email address for the observer to process")
    semester : typing.Optional[str] = xconf.field(default=None, help="Semester to search in, 202XXA/20XXB format")
    utc_start : typing.Optional[datetime.datetime] = xconf.field(default=None, help="ISO UTC datetime stamp of earliest observation start time to process (supersedes semester)")
    utc_end : typing.Optional[datetime.datetime] = xconf.field(default=None, help="ISO UTC datetime stamp of latest observation end time to process (supersedes semester)")
    data_roots : list[str] = xconf.field(default_factory=constants.LOOKYLOO_DATA_ROOTS.copy, help=f"Search directory for telem and rawimages subdirectories, repeat to specify multiple roots. (default: {constants.LOOKYLOO_DATA_ROOTS})")

    def main(self):
        raise NotImplementedError("Command subclasses must implement main()")

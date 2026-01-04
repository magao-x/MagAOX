import datetime
from datetime import timezone
import socket
from pathlib import Path
import typing
import psycopg
import xconf

import magaox.db.config as dbconfig

from ... import constants, utils

from ..core import (
    PathRewriteConfig,
)


def generate_path_rewrites():
    return [
        PathRewriteConfig(hostname='exao2', from_path='/opt/MagAOX/calib', to_path='/srv/rtc/calib'),
        PathRewriteConfig(hostname='exao2', from_path='/opt/MagAOX', to_path='/srv/rtc/data'),
        PathRewriteConfig(hostname='exao3', from_path='/opt/MagAOX/calib', to_path='/srv/icc/calib'),
        PathRewriteConfig(hostname='exao3', from_path='/opt/MagAOX', to_path='/srv/icc/data'),
    ]

@xconf.config
class BaseQuicklookCommand(dbconfig.BaseConfig, xconf.Command):
    target : typing.Optional[str] = xconf.field(default=None, help="The target name (as entered by the observer)")
    title : typing.Optional[str] = xconf.field(default=None, help="All or part of the observation name to process")
    exact_title : bool = xconf.field(default=False, help="Whether to match the given title as a substring anywhere in the observation title")
    email : typing.Optional[str] = xconf.field(default=None, help="Email address for the observer to process")
    semester : typing.Optional[str] = xconf.field(default=utils.get_current_semester(), help="Semester to search in, 202XXA/20XXB format")
    utc_start : typing.Optional[datetime.datetime] = xconf.field(default=None, help="ISO UTC datetime stamp of earliest observation start time to process (supersedes semester)")
    utc_end : typing.Optional[datetime.datetime] = xconf.field(default=None, help="ISO UTC datetime stamp of latest observation end time to process (supersedes semester)")
    data_roots : list[Path] = xconf.field(default_factory=constants.LOOKYLOO_DATA_ROOTS.copy, help=f"Search directory for telem and rawimages subdirectories, repeat to specify multiple roots. (default: {constants.LOOKYLOO_DATA_ROOTS})")
    common_path_prefix : str = xconf.field(default=constants.DEFAULT_PREFIX, help="Prefix for all instrument data and config directories")
    temp_root : str = xconf.field(default='/data/users/xsup/temp', help="Temporary folder on disk for array storage during re-packing")
    path_rewrites : list[PathRewriteConfig] = xconf.field(default_factory=generate_path_rewrites, help="Rewrite the paths in the inventory (e.g. to use an NFS mount to read from another host)")

    def get_time_range(self):
        start_dt, end_dt = utils.semester_to_datetime_range(self.semester if self.semester is not None else utils.get_current_semester())
        if self.utc_start is not None:
            start_dt = self.utc_start
        if self.utc_end is not None:
            end_dt = self.utc_end
        return start_dt, end_dt

    def main(self):
        raise NotImplementedError("Command subclasses must implement main()")

import datetime
from dataclasses import dataclass
import pathlib
from typing import Union

@dataclass
class FileOrigin:
    origin_host : str
    origin_path : Union[pathlib.Path, str]
    mtime : datetime.datetime
    size_bytes : int

@dataclass
class FileReplica:
    origin_host : str
    origin_path : Union[pathlib.Path, str]
    origin_mtime : datetime.datetime
    size_bytes : int
    hostname : str
    replica_path : Union[pathlib.Path, str]

@dataclass
class Telem:
    # pretty-printed log/teldump row
    # {
    #     "ts": "2024-04-15T22:30:50.997960288",
    #     "prio": "TELM",
    #     "ec": "telem_telpos",
    #     "msg": {
    #         "epoch": 0.0,
    #         "ra": 111.575541666667,
    #         "dec": -29.013666666667,
    #         "el": 89.9997,
    #         "ha": 0.000833333333,
    #         "am": 1.0,
    #         "rotoff": 77.7293
    #     }
    # }
    #
    # The "prio" is always "TELM" so we can ignore it. "ec" is telemetry type.
    # "msg" is freeform / semi-structured (consistent within an "ec")
    # device is not self-documenting, only known from the filename that produced the
    # row.
    device : str
    ts : datetime.datetime
    ec : str
    msg : dict


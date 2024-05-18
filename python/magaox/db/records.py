import orjson
import datetime
from dataclasses import dataclass
import pathlib
from typing import Union

from ..utils import parse_iso_datetime_as_utc, FunkyJSONDecoder

__all__ = [
    'FileOrigin',
    'FileReplica',
    'FileIngestTime',
    'Telem',
]

FALLBACK_JSON = FunkyJSONDecoder()

@dataclass
class FileOrigin:
    origin_host : str
    origin_path : Union[pathlib.Path, str]
    creation_time : datetime.datetime
    modification_time : datetime.datetime
    size_bytes : int

@dataclass
class FileReplica:
    origin_host : str
    origin_path : Union[pathlib.Path, str]
    origin_modification_time : datetime.datetime
    size_bytes : int
    hostname : str
    replica_path : Union[pathlib.Path, str]

@dataclass
class FileIngestTime:
    ts : datetime.datetime
    device : str
    ingested_at : datetime.datetime
    origin_host : str
    origin_path : str

def _parse_msg_json(msg_json: str):
    msg_bytes = msg_json.encode('utf8')
    try:
        payload = orjson.loads(msg_bytes)
    except orjson.JSONDecodeError:
        payload = FALLBACK_JSON.decode(msg_json)
    return payload

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

    @classmethod
    def from_json(cls, device, json_str):
        payload = _parse_msg_json(json_str)
        return cls(
            device=device,
            ts=parse_iso_datetime_as_utc(payload['ts']),
            ec=payload['ec'],
        )

    def get_msg_json_bytes(self) -> bytes:
        return orjson.dumps(self.msg)

    def get_msg_json(self) -> str:
        return self.get_msg_json_bytes().decode('utf8')
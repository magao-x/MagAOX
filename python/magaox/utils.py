import os.path
import datetime
from datetime import timezone

__all__ = [
    'XFILENAME_TIME_FORMAT',
    'PUREPYINDI_DEVICE_FILENAME_TIME_FORMAT',
    'parse_iso_datetime_as_utc',
    'xfilename_to_utc_timestamp',
    'creation_time_from_filename',
    'parse_iso_datetime',
    'utcnow',
    'format_timestamp_for_filename',
    'get_current_semester',
    'get_search_start_end_timestamps',
    'FunkyJSONDecoder',
]

# note: we must truncate to microsecond precision due to limitations in
# `datetime`, so this pattern works only after chopping off the last
# three characters
XFILENAME_TIME_FORMAT = "%Y%m%d%H%M%S%f"
# Python devices use a different time stamp format
PUREPYINDI_DEVICE_FILENAME_TIME_FORMAT = "%Y-%m-%dT%H%M%S"

def parse_iso_datetime_as_utc(input_str):
    input_str = input_str[:26]  # chop off nanoseconds and anything else
    dt = datetime.datetime.fromisoformat(input_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt

def xfilename_to_utc_timestamp(filename):
    _, filename = os.path.split(filename)
    name, rest = filename.rsplit('_', 1)
    chopped_ts_str, ext = rest.split('.', 1)
    try:
        # nanoseconds are too precise for Python's native datetimes
        ts = datetime.datetime.strptime(chopped_ts_str[:-3], XFILENAME_TIME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        ts =  datetime.datetime.strptime(chopped_ts_str, PUREPYINDI_DEVICE_FILENAME_TIME_FORMAT).replace(tzinfo=timezone.utc)
    return ts

def creation_time_from_filename(filepath, stat_result=None):
    try:
        ts = xfilename_to_utc_timestamp(filepath)
    except ValueError:
        if stat_result is None:
            stat_result = os.stat(filepath)
        ts = datetime.datetime.fromtimestamp(stat_result.st_ctime)
    return ts


import typing
import datetime
from datetime import timezone

from .constants import FOLDER_TIMESTAMP_FORMAT

def parse_iso_datetime(input_str):
    dt = datetime.datetime.fromisoformat(input_str)
    if input_str[-1] == 'Z':
        input_str = input_str[:-1]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def utcnow():
    return datetime.datetime.utcnow().replace(tzinfo=timezone.utc)


def format_timestamp_for_filename(ts):
    return ts.strftime(FOLDER_TIMESTAMP_FORMAT)

def get_current_semester():
    now = datetime.datetime.now()
    this_year = now.year
    this_semester = str(this_year) + ("B" if now.month > 6 else "A")
    return this_semester

def get_search_start_end_timestamps(
    semester : str,
    utc_start : typing.Optional[datetime.datetime] = None,
    utc_end : typing.Optional[datetime.datetime] = None,
):
    letter = semester[-1].upper()
    try:
        if len(semester) != 5 or semester[-1].upper() not in ['A', 'B']:
            raise ValueError()
        year = int(semester[:-1])
        month = 1 if letter == 'A' else 6
        day = 15 if month == 6 else 1
    except ValueError:
        raise RuntimeError(f"Got {semester=} but need a 4 digit year + A or B (e.g. 2022A)")
    semester_start_dt = datetime.datetime(year, month, day)
    semester_start_dt = semester_start_dt.replace(tzinfo=timezone.utc)
    start_dt = semester_start_dt
    semester_end_dt = datetime.datetime(
        year=year + 1 if letter == 'B' else year,
        month=1 if letter == 'B' else 6,
        day = 15 if letter == 'A' else 1,
    ).replace(tzinfo=timezone.utc)
    end_dt = semester_end_dt


    if utc_start is not None:
        start_dt = utc_start

    if utc_end is not None:
        end_dt = utc_end

    if end_dt < start_dt:
        raise ValueError("End time is before start time")
    return start_dt, end_dt


import json

from json.scanner import NUMBER_RE

def py_make_scanner(context):
    '''Derived from the Python standard library json.scanner module
    used under the terms of the Python software license

    https://docs.python.org/3/license.html#psf-license-agreement-for-python-release

    The function has been modified to handle literal ``nan``, ``-nan``, ``inf``, ``-inf``
    as emitted by Flatbuffers "JSON" support
    '''
    parse_object = context.parse_object
    parse_array = context.parse_array
    parse_string = context.parse_string
    match_number = NUMBER_RE.match
    strict = context.strict
    parse_float = context.parse_float
    parse_int = context.parse_int
    parse_constant = context.parse_constant
    object_hook = context.object_hook
    object_pairs_hook = context.object_pairs_hook
    memo = context.memo

    def _scan_once(string, idx):
        try:
            nextchar = string[idx]
        except IndexError:
            raise StopIteration(idx) from None

        if nextchar == '"':
            return parse_string(string, idx + 1, strict)
        elif nextchar == '{':
            return parse_object((string, idx + 1), strict,
                _scan_once, object_hook, object_pairs_hook, memo)
        elif nextchar == '[':
            return parse_array((string, idx + 1), _scan_once)
        elif nextchar == 'n' and string[idx:idx + 4] == 'null':
            return None, idx + 4
        elif nextchar == 't' and string[idx:idx + 4] == 'true':
            return True, idx + 4
        elif nextchar == 'f' and string[idx:idx + 5] == 'false':
            return False, idx + 5

        m = match_number(string, idx)
        if m is not None:
            integer, frac, exp = m.groups()
            if frac or exp:
                res = parse_float(integer + (frac or '') + (exp or ''))
            else:
                res = parse_int(integer)
            return res, m.end()
        elif nextchar == 'N' and string[idx:idx + 3] == 'NaN':
            return parse_constant('NaN'), idx + 3
        elif nextchar == 'I' and string[idx:idx + 8] == 'Infinity':
            return parse_constant('Infinity'), idx + 8
        elif nextchar == '-' and string[idx:idx + 9] == '-Infinity':
            return parse_constant('-Infinity'), idx + 9
        elif nextchar == 'n' and string[idx:idx + 3] == 'nan':
            return parse_constant('NaN'), idx + 3
        elif nextchar == '-' and string[idx:idx + 4] == '-nan':
            return parse_constant('NaN'), idx + 4
        elif nextchar == 'i' and string[idx:idx + 3] == 'inf':
            return parse_constant('Infinity'), idx + 3
        elif nextchar == '-' and string[idx:idx + 4] == '-inf':
            return parse_constant('-Infinity'), idx + 4
        else:
            raise StopIteration(idx)

    def scan_once(string, idx):
        try:
            return _scan_once(string, idx)
        finally:
            memo.clear()

    return scan_once

class FunkyJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_once = py_make_scanner(self)

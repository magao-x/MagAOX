import os.path
import datetime
from datetime import timezone

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

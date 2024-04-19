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
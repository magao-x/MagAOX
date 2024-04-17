import datetime

def parse_iso_datetime_as_utc(input_str):
    input_str = input_str[:26]  # chop off nanoseconds and anything else
    dt = datetime.datetime.fromisoformat(input_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt
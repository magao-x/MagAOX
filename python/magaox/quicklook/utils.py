
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
    return datetime.datetime.now(timezone.utc)


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

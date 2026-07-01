import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import polars as pl


# Batch UTC time embedded in ``group_suffix`` / distill ``suffix`` (see ``realtime.format_batch_timestamp``).
# Digit-delimited (not ``\\b``) so a trailing underscore after microseconds still matches.
_BATCH_UTC_TOKEN_RE = re.compile(r"(?<!\d)(\d{8}T\d{6}\d{6})(?!\d)")
# Compact variant without ``T`` (and optional ``_00000`` trailer), e.g. ``camwfs_20230313071857408144000``.
_BATCH_UTC_COMPACT_RE = re.compile(r"(?<!\d)(\d{8})(\d{6})(\d*)(?!\d)")


def _parse_iso_timestamp_to_utc_seconds(ts_str):
    """
    Parse an ISO-like timestamp string to Unix seconds (UTC).

    Supports nanosecond (or arbitrary-length) fractional seconds; ``datetime.fromisoformat``
    only accepts up to 6 fractional digits, so sub-microsecond tails are parsed manually.
    Naive times are treated as UTC. Optional trailing ``Z`` or ``±HH:MM`` offsets are honored.
    """
    text = str(ts_str).strip()
    if not text:
        raise ValueError("empty timestamp")
    text = text.replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    m_tz = re.search(r"([+-])(\d{2}):(\d{2})$", text)
    if m_tz:
        sign = 1 if m_tz.group(1) == "+" else -1
        offset_sec = sign * (int(m_tz.group(2)) * 3600 + int(m_tz.group(3)) * 60)
        core = text[: m_tz.start()]
        tzinfo = timezone(timedelta(seconds=offset_sec))
    else:
        core = text
        tzinfo = timezone.utc

    if "." in core:
        main, rest = core.split(".", 1)
        digits = "".join(c for c in rest if c.isdigit())
        if digits:
            frac = Decimal(digits) / (Decimal(10) ** len(digits))
        else:
            frac = Decimal(0)
    else:
        main = core
        frac = Decimal(0)

    dt = datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tzinfo)
    return dt.timestamp() + float(frac)

def parse_batch_utc_seconds_from_suffix(suffix):
    """
    Parse the UTC batch instant embedded in ``suffix``.

    Supports:

    - ``YYYYMMDDTHHMMSSffffff`` as produced by ``realtime.format_batch_timestamp``.
    - Compact ``YYYYMMDDHHMMSS`` + optional fractional digits (no ``T``), e.g.
      ``camwfs_20230313071857408144000`` (fractional tail interpreted as
      ``int(frac) / 10**len(frac)`` seconds).
    """
    m = _BATCH_UTC_TOKEN_RE.search(suffix)
    if m:
        token = m.group(1)
        dt = datetime.strptime(token, "%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc)
        return dt.timestamp()

    # Use the rightmost compact match when multiple digit runs appear in ``suffix``.
    matches = list(_BATCH_UTC_COMPACT_RE.finditer(suffix))
    if not matches:
        raise ValueError(
            "Could not find batch UTC time in distill suffix (expected "
            "YYYYMMDDTHHMMSSffffff or YYYYMMDDHHMMSS plus optional fractional digits): "
            f"{suffix!r}"
        )
    m2 = matches[-1]
    ymd, hms, frac = m2.group(1), m2.group(2), m2.group(3)
    if len(ymd) != 8 or len(hms) != 6:
        raise ValueError(f"Invalid compact batch date/time in suffix: {suffix!r}")
    base = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if not frac:
        return base.timestamp()
    subsec = int(frac, 10) / (10.0 ** len(frac))
    return base.timestamp() + subsec

def normalize_cube_timestamp_compact(current_time: str) -> str:
    """Return ``YYYYMMDDHHMMSSffffff`` suitable for ``datetime.strptime`` ``%Y%m%d%H%M%S%f``.

    Realtime cubes may use an ISO-like middle token ``20260411T020434811`` (``T`` between
    date and time). Older distill outputs used a single digit run
    ``20230310054802555791000``; the last three digits are stripped as nanoseconds so the
    remainder parses as microseconds.
    """
    if "T" in current_time:
        date_part, rest = current_time.split("T", 1)
        if len(rest) < 6:
            raise ValueError(
                f"Invalid cube timestamp (expected HHMMSS after 'T'): {current_time!r}"
            )
        hhmmss = rest[:6]
        frac = rest[6:]
        if not frac:
            micro = "000000"
        elif len(frac) <= 6:
            micro = (frac + "000000")[:6]
        else:
            micro = frac[:6]
        return f"{date_part}{hhmmss}{micro}"
    return current_time[:-3]


def are_we_past_transit(current_time, transit_time):
    # Transit from config, e.g. 2023-03-10T06:09:11.342216344Z
    s_transit = pl.Series(name="transit_time", values=[transit_time])
    dt_transit = s_transit.str.to_datetime(
        "%Y-%m-%dT%H:%M:%S%.9fZ", time_zone="UTC"
    )
    compact = normalize_cube_timestamp_compact(current_time)
    dt_current = datetime.strptime(compact, "%Y%m%d%H%M%S%f").replace(
        tzinfo=timezone.utc
    )
    past_transit = dt_current > dt_transit.item()

    return past_transit
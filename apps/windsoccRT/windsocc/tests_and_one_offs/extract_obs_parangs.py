#!/usr/bin/env python3
"""
One-off: read PARANG and timestamps from camsci FITS headers under a raw data directory.

TODO: is there a way to get the PARANG from the camwfs FITS headers? How would
this work in real-time mode?


Writes PARANG, timestamp, and seconds elapsed since the earliest timestamp in the batch.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from astropy.io import fits

# Location of the raw science data
# raw_science_data_dir = "/Volumes/FantomHD/magaox_data/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci1"
# raw_science_data_dir = "/Users/jkueny/data/HR4796_rg_smlyot_20230312_13/unsats_rg_20230313T071732/camsci1"
# raw_science_data_dir = "/Volumes/FantomHD/magaox_data/HR4796_rg_smlyot_20230312_13/raws_rg_smlyot_20230313T043914/camsci1"
raw_science_data_dir = "/Volumes/FantomHD/magaox_data/HR4796_rg_smlyot_20230312_13/unsats_rg_20230313T071732/camsci1"
# Output path (TSV: PARANG, timestamp, elapsed_seconds)
# save_obs_parangs_here = "/Users/jkueny/data/HR4796_rg_smlyot_20230312_13/unsats_rg_20230313T071732/parangs.txt"
# save_obs_parangs_here = "/Volumes/FantomHD/magaox_data/HR4796_rg_smlyot_20230312_13/raws_rg_smlyot_20230313T043914/parangs.txt"
save_obs_parangs_here = "/Volumes/FantomHD/magaox_data/HR4796_rg_smlyot_20230312_13/unsats_rg_20230313T071732/parangs.txt"

_ISO_DATE_HEADER_KEYS = ("DATE-OBS", "DATE", "DATE_OBS")
# FITS DATE-* values sometimes include fractional seconds or a trailing Z.
_ISO_T_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


def _format_header_timestamp(value: object) -> str | None:
    """Normalize a header date string to ``YYYY-mm-ddTHH:MM:SS`` or with fractional seconds."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = _ISO_T_RE.match(text.replace(" ", "T", 1))
    if m:
        base, frac, _tz = m.group(1), m.group(2), m.group(3)
        if frac:
            return base + frac
        return base
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt.microsecond:06d}".rstrip("0").rstrip(".")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _timestamp_from_header(header: fits.Header) -> str | None:
    for key in _ISO_DATE_HEADER_KEYS:
        raw = header.get(key)
        if raw in (None, "", " "):
            continue
        formatted = _format_header_timestamp(raw)
        if formatted:
            return formatted
    return None


def _timestamp_from_filename(path: Path) -> str | None:
    """
    Fallback: stem split on ``_``; index 1 is ``YYYYMMDDHHMMSS`` plus optional
    nanosecond digits (variable length).
    """
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    token = parts[1]
    if len(token) < 14:
        return None
    ymd_hms = token[:14]
    try:
        datetime.strptime(ymd_hms, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    base = (
        f"{ymd_hms[0:4]}-{ymd_hms[4:6]}-{ymd_hms[6:8]}"
        f"T{ymd_hms[8:10]}:{ymd_hms[10:12]}:{ymd_hms[12:14]}"
    )
    frac = token[14:]
    if not frac:
        return base
    frac_digits = "".join(ch for ch in frac if ch.isdigit())
    if not frac_digits:
        return base
    # Preserve up to 9 fractional digits (nanoseconds) as a decimal tail.
    frac_digits = frac_digits[:9].ljust(9, "0")
    return f"{base}.{frac_digits}".rstrip("0").rstrip(".")


def _parse_instant(ts: str) -> tuple[datetime, Decimal] | None:
    """
    Parse a normalized timestamp into a naive wall-clock ``datetime`` plus a
    fractional-second part in ``[0, 1)``.
    """
    ts = ts.strip()
    if not ts:
        return None
    if "." in ts:
        main, rest = ts.split(".", 1)
        digits = "".join(c for c in rest if c.isdigit())
        if not digits:
            frac = Decimal(0)
        else:
            frac = Decimal(digits) / (Decimal(10) ** len(digits))
    else:
        main = ts
        frac = Decimal(0)
    try:
        dt = datetime.strptime(main, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return (dt, frac)


def _elapsed_since(
    t0: tuple[datetime, Decimal], t1: tuple[datetime, Decimal]
) -> float:
    """Seconds from ``t0`` to ``t1`` (naive instants, subsecond via ``Decimal``)."""
    dt0, f0 = t0
    dt1, f1 = t1
    return (dt1 - dt0).total_seconds() + float(f1 - f0)


def _parang_and_timestamp(path: Path) -> tuple[str, str] | None:
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        if "PARANG" not in hdr:
            return None
        parang = hdr["PARANG"]
        if hasattr(parang, "value"):
            parang = parang.value
        ts = _timestamp_from_header(hdr) or _timestamp_from_filename(path)
        if ts is None:
            return None
        return (str(parang), ts)


def main() -> None:
    data_dir = Path(raw_science_data_dir)
    out_path = Path(save_obs_parangs_here)
    fits_paths = sorted(data_dir.glob("camsci*.fits"))
    rows: list[tuple[str, str]] = []
    for p in fits_paths:
        got = _parang_and_timestamp(p)
        if got is None:
            print(f"skip (missing PARANG or timestamp): {p.name}")
            continue
        rows.append(got)

    instants: list[tuple[datetime, Decimal] | None] = [
        _parse_instant(ts) for _parang, ts in rows
    ]
    if any(i is None for i in instants):
        bad = [rows[j] for j, i in enumerate(instants) if i is None]
        raise ValueError(f"Could not parse timestamps for rows: {bad!r}")
    parsed = [i for i in instants if i is not None]
    t0 = min(parsed) if parsed else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("PARANG\ttimestamp\telapsed_seconds\n")
        if t0 is not None:
            for (parang, ts), inst in zip(rows, parsed):
                elapsed = _elapsed_since(t0, inst)
                f.write(f"{parang}\t{ts}\t{elapsed:g}\n")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

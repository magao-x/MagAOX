#!/usr/bin/env python3
"""
Quick one-off: read DATE-OBS from FITS headers, sort by time, report frame spacing stats.
"""

from pathlib import Path

from astropy.io import fits
from astropy.time import Time


FITS_DIR = Path(
    "/Users/jkueny/data/HR4796_rg_smlyot_20230312_13/raws_rg_smlyot_20230313T043914/camsci1/lite_wdhlib"
)
FITS_GLOB = "*.fits"


def main():
    paths = sorted(FITS_DIR.glob(FITS_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matching '{FITS_GLOB}' under {FITS_DIR}")

    rows = []
    for path in paths:
        with fits.open(path) as hdul:
            date_obs = hdul[0].header.get("DATE-OBS")
        if date_obs is None:
            raise KeyError(f"DATE-OBS missing in {path}")
        rows.append((Time(date_obs, format="isot", scale="utc"), path))

    rows.sort(key=lambda r: r[0])
    times = [r[0] for r in rows]

    if len(times) < 2:
        print(f"Only {len(times)} FITS file(s); need at least two for deltas.")
        return

    deltas_sec = [(times[i + 1] - times[i]).sec for i in range(len(times) - 1)]
    deltas_rounded = [round(float(d)) for d in deltas_sec[2:-2]]

    avg_delta = sum(deltas_rounded) / len(deltas_rounded)
    p2p = max(deltas_rounded) - min(deltas_rounded)

    print(f"Directory: {FITS_DIR}")
    print(f"FITS count: {len(paths)}")
    print(f"Successive time deltas (seconds, rounded): {deltas_rounded}")
    print(f"Average delta (s): {avg_delta:.6g}")
    print(f"Peak-to-peak delta (max-min of successive deltas, s): {p2p}")


if __name__ == "__main__":
    main()

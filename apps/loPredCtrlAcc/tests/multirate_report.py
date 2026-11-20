#!/usr/bin/env python3
"""T9 PSD report: compare 8-kHz residual spectra of standard and assisted DDSPC by ACC information level.

Reads <case>_<mode>.f32 files written by the "[multirate-sim]" test when DDSPC_MULTIRATE_TRACE names a directory:
float32 X/Y residual pairs at the 8-kHz slot rate. Prints the variance and the RMS in frequency bands, and with
--plot writes a PNG of the Welch PSDs.

    mkdir -p /tmp/t9 && DDSPC_MULTIRATE_TRACE=/tmp/t9 build/offline/rtc-controller/ddspc_offline_test '[multirate-sim]'
    python3 tests/multirate_report.py /tmp/t9 [--plot /tmp/t9/psd.png]
"""
import argparse
import pathlib

import numpy as np
from scipy import signal

FS = 8000.0
BANDS = [(0, 100), (100, 250), (250, 400), (400, 1000), (1000, 4000)]
CASES = ["strong", "weak", "zero", "disconnected"]


def psd(path: pathlib.Path):
    """Return frequencies and the X+Y Welch PSD of one trace."""
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 2)
    f, p = signal.welch(data, fs=FS, window="hann", nperseg=min(4096, len(data)), axis=0)
    return f, p.sum(axis=1), float(np.mean(np.sum(data.astype(np.float64) ** 2, axis=1)))


def band_rms(f, p, low, high):
    selected = (f >= low) & (f < high)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(np.sqrt(integrate(p[selected], f[selected]))) if selected.any() else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--plot", type=pathlib.Path)
    args = parser.parse_args()
    header = "case          mode      variance " + " ".join(f"{lo:>4d}-{hi:<4d}Hz" for lo, hi in BANDS)
    print(header)
    spectra = {}
    for case in CASES:
        for mode in ("standard", "assisted"):
            path = args.directory / f"{case}_{mode}.f32"
            if not path.exists():
                continue
            f, p, variance = psd(path)
            spectra[case, mode] = (f, p)
            bands = " ".join(f"{band_rms(f, p, lo, hi):10.4f}" for lo, hi in BANDS)
            print(f"{case:13s} {mode:9s} {variance:8.4f} {bands}")
    if args.plot and spectra:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cases = [c for c in CASES if (c, "standard") in spectra]
        fig, axes = plt.subplots(len(cases), 1, figsize=(8, 2.6 * len(cases)), sharex=True)
        for axis, case in zip(np.atleast_1d(axes), cases):
            for mode in ("standard", "assisted"):
                if (case, mode) in spectra:
                    f, p = spectra[case, mode]
                    axis.loglog(f[1:], p[1:], label=mode)
            axis.set_title(f"ACC information: {case}")
            axis.set_ylabel("PSD [unit²/Hz]")
            axis.legend()
        np.atleast_1d(axes)[-1].set_xlabel("Frequency [Hz]")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()

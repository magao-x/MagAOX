#!/usr/bin/env python3
"""Export an OCAM manual reconfigure history to CSV."""

import argparse
import csv
import struct
import sys
from pathlib import Path

MAGIC = b"OCAMFD01"
HEADER = struct.Struct("<9Q")
EPOCH = struct.Struct("<QQQdQQQ")
ROW = struct.Struct("<qqqQQII")


def export(source):
    """Validate a version 1 dump and stream its rows with the matching epoch."""
    with source.open("rb") as binary:
        if binary.read(len(MAGIC)) != MAGIC:
            raise ValueError("unrecognized frame history format")
        request, cutoff, realtime, boundary, dropped, trimmed, rows, epochs, duration = (
            HEADER.unpack(binary.read(HEADER.size))
        )
        expected = len(MAGIC) + HEADER.size + epochs * EPOCH.size + rows * ROW.size
        if source.stat().st_size != expected or (rows and not epochs):
            raise ValueError("incorrect frame history size or missing epoch")
        context = [EPOCH.unpack(binary.read(EPOCH.size)) for _ in range(epochs)]
        print(
            f"request={request} cutoff_ns={cutoff} realtime_ns={realtime} "
            f"boundary={boundary} dropped={dropped} trimmed={trimmed} "
            f"rows={rows} duration_ns={duration}",
            file=sys.stderr,
        )
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(
            [
                "sequence", "atime_ns", "arrival_ns", "write_ns", "counter", "frame", "outcome",
                "possible_clock_step", "session", "inode", "fps", "mode", "bin_x", "bin_y",
            ]
        )
        index = 0
        for _ in range(rows):
            atime, arrival, write, sequence, counter, frame, flags = ROW.unpack(binary.read(ROW.size))
            while index + 1 < len(context) and context[index + 1][0] <= sequence:
                index += 1
            writer.writerow(
                [
                    sequence, atime if atime >= 0 else "", arrival if flags & 16 else "",
                    write if write >= 0 else "", counter if flags & 8 else "", frame if flags & 4 else "",
                    ("published", "rejected", "failed", "unknown")[flags & 3],
                    bool(flags & 32), *context[index][1:],
                ]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="camera frame-delay .bin snapshot")
    args = parser.parse_args()
    try:
        export(args.file)
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

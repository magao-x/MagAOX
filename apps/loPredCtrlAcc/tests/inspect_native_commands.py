r"""\file inspect_native_commands.py
\brief Read the private full-command journal; publication records never certify DM application.
"""

import argparse
import json
import math
from pathlib import Path
import struct


HEADER = struct.Struct("<6Q4q7Q")


def inspect(path, allow_partial=False, values=8):
    """Stream complete records without modifying the file or allocating unbounded payloads."""
    count, invalid, first, latest = 0, 0, None, None
    partial = False
    with Path(path).open("rb") as stream:
        magic = stream.read(8)
        if magic != b"DDDM0001":
            raise ValueError("Missing/unsupported command journal header")
        while True:
            header = stream.read(HEADER.size)
            if not header:
                break
            if len(header) != HEADER.size:
                partial = True
                break
            words = HEADER.unpack(header)
            elements, device, file_inode, image_inode, generation, counter = words[:6]
            acquired, published, received, available = words[6:10]
            width, height, depth, datatype, flags, digest, counter1 = words[10:]
            if not 1 <= elements <= 65536 or width * height != elements or depth < 1:
                raise ValueError("Invalid command payload extent")
            payload = stream.read(elements * 4)
            if len(payload) != elements * 4:
                partial = True
                break
            command = struct.unpack(f"<{elements}f", payload)
            invalid += sum(not math.isfinite(value) for value in command)
            latest = dict(device=device, file_inode=file_inode, image_inode=image_inode,
                          generation=generation, counter=counter, counter1=counter1,
                          acquired_raw_ns=acquired, published_raw_ns=published,
                          received_mono_ns=received, available_mono_ns=available,
                          width=width, height=height, depth=depth, datatype=datatype,
                          flags=flags, payload_hash=digest, elements=elements,
                          first_values=[value if math.isfinite(value) else None for value in command[:values]])
            first = available if first is None else first
            count += 1
    if partial and not allow_partial:
        raise ValueError("Truncated command record; use --allow-partial only while the observer is running")
    return dict(records=count, nonfinite_values=invalid, incomplete_tail=partial,
                received_rate_hz=(count-1)*1e9/(latest["available_mono_ns"]-first)
                if count > 1 and latest["available_mono_ns"] > first else None,
                evidence="command_publication_only", physical_application_qualified=False,
                latest=latest)


def main():
    """Print retained values and exact timestamp/counter identities from an operator-selected local journal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--values", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.values <= 65536:
        parser.error("--values must be in [0,65536]")
    try:
        result = inspect(args.journal, args.allow_partial, args.values)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result["nonfinite_values"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())

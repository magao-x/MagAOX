r"""\file compare_profile_traces.py
\brief Compare raw per-version profile matrices; exact mode rejects any changed numerical byte.
"""
import argparse
import json
import math
import struct


def records(path):
    """Decode the fixed native little-endian offline trace without reconstructing a DDSPC algorithm."""
    with open(path, "rb") as stream:
        while True:
            header = stream.read(64)
            if not header:
                return
            assert len(header) == 64, "Truncated record header"
            values = struct.unpack("<8Q", header)
            matrices = [stream.read(size * 4) for size in values[4:]]
            rmse = stream.read(8)
            assert all(len(data) == size * 4 for data, size in zip(matrices, values[4:])), "Truncated matrix"
            assert len(rmse) == 8, "Truncated RMSE"
            yield values, matrices + [rmse]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    report = {}
    left, right = iter(records(args.before)), iter(records(args.after))
    exact = True
    while True:
        a, b = next(left, None), next(right, None)
        if a is None or b is None:
            assert a is None and b is None, "Trace lengths differ"
            break
        assert a[0] == b[0], "Record identity or matrix shape differs"
        key = f"m{a[0][0]}_{'qr' if a[0][1] else 'rls'}_{a[0][2]}"
        item = report.setdefault(key, {"records": 0, "changed_bytes": 0,
                                      "max_theta": 0, "max_gain": 0, "max_state": 0,
                                      "max_proposal": 0, "max_rmse": 0})
        item["records"] += 1
        for name, before, after in zip(("theta", "gain", "state", "proposal", "rmse"), a[1], b[1]):
            assert len(before) == len(after)
            if before == after:
                continue
            exact = False
            item["changed_bytes"] += sum(x != y for x, y in zip(before, after))
            kind = "d" if name == "rmse" else "f"
            size = 8 if name == "rmse" else 4
            x, y = struct.unpack(f"<{len(before)//size}{kind}", before), struct.unpack(f"<{len(after)//size}{kind}", after)
            differences = [abs(v - w) if math.isfinite(v) and math.isfinite(w) else math.inf
                           for v, w in zip(x, y)]
            item["max_" + name] = max(item["max_" + name], max(differences))
    assert report, "Empty traces do not establish numerical equivalence"
    print(json.dumps({"byte_exact": exact, "profiles": report}, indent=2))
    if args.require_exact and not exact:
        raise SystemExit(1)

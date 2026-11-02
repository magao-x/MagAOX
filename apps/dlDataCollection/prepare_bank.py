#!/usr/bin/env python3
"""Prepare validated dlDataCollection banks offline, without connecting to a DM."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def checksum(path):
    """Return SHA-256 without loading another copy of a large payload."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_mask(path):
    """Load an explicit binary 50x50 actuator mask; never invent excluded actuators."""
    path = Path(path)
    if path.suffix == ".npy":
        mask = np.load(path, allow_pickle=False)
    elif path.suffix == ".bin":
        mask = np.fromfile(path, dtype="<f4")
    else:
        mask = np.loadtxt(path, delimiter=",")
    if mask.size != 2500 or not np.all(np.isfinite(mask)):
        raise ValueError("The recorded mask must have exactly 2500 finite values.")
    mask = mask.reshape(2500)
    if not np.all((mask == 0) | (mask == 1)) or not np.any(mask == 1):
        raise ValueError("The mask must contain binary values and at least one active actuator.")
    return mask.astype("<f4")


def validate_commands(commands, mask, samples):
    """Reject malformed, non-finite, or excluded-actuator commands without clipping."""
    commands = np.asarray(commands)
    if commands.size != samples * 2500:
        raise ValueError(f"Expected {samples * 2500} coefficients, got {commands.size}.")
    commands = commands.reshape(samples, 2500).astype("<f4")
    if not np.all(np.isfinite(commands)):
        raise ValueError("All applied float32 coefficients must be finite.")
    if np.any(commands[:, mask == 0] != 0):
        raise ValueError("Excluded actuators are nonzero; regenerate or explicitly request --repair-mask.")
    return commands


def write_payload(path, values):
    """Write an exclusive temporary payload, flush it, then publish it in a new bank."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as stream:
        values.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    if temporary.stat().st_size != values.nbytes:
        raise OSError(f"Incomplete float32 payload: {temporary}")
    os.link(temporary, path)
    temporary.unlink()
    return checksum(path)


def write_bank(output, mask, samples, sets, commands, provenance):
    """Publish the manifest last; incomplete or existing banks cannot be acquired."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mask_hash = write_payload(output / "actuator_mask.bin", mask.astype("<f4"))
    fields = [
        "format=dlDataCollection-1",
        "dtype=float32-le",
        "units=um_surface",
        "modes=2500",
        f"samples={samples}",
        f"sets={sets}",
        f"mask_sha256={mask_hash}",
    ]
    count = 0
    for index, values in enumerate(commands):
        if index >= sets:
            raise ValueError("Too many coefficient sets.")
        values = validate_commands(values, mask, samples)
        value_hash = write_payload(output / f"modeval_dataset_{index}.bin", values)
        fields.append(f"set_{index}_sha256={value_hash}")
        count += 1
    if count != sets:
        raise ValueError(f"Expected {sets} sets, got {count}.")
    provenance = dict(provenance, numpy_version=np.__version__, script_sha256=checksum(__file__))
    with (output / "generation.json").open("x") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fields.append(f"generation_sha256={checksum(output / 'generation.json')}")
    with (output / "manifest.conf").open("x") as stream:
        stream.write("\n".join(fields) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def random_commands(mask, samples, sets, seed, ptv_min, ptv_max, exponent_min, exponent_max):
    """Generate every sample with a seeded HCIPy surface-error spectrum and masked piston."""
    from hcipy import Field, SpectralNoiseFactoryFFT, make_circular_aperture, make_pupil_grid

    grid = make_pupil_grid(50, 1)
    aperture = make_circular_aperture(1)(grid) != 0
    active = mask != 0
    rng = np.random.default_rng(seed)
    for _ in range(sets):
        values = np.empty((samples, 2500), dtype="<f4")
        peaks = 10 ** rng.uniform(math.log10(ptv_min), math.log10(ptv_max), samples)
        exponents = rng.uniform(exponent_min, exponent_max, samples)
        for index in range(samples):
            exponent = exponents[index]

            def psd(frequency_grid):
                radius = frequency_grid.as_("polar").r
                result = Field(np.zeros(radius.shape), frequency_grid)
                good = radius != 0
                result[good] = radius[good] ** exponent
                return result

            surface = np.asarray(SpectralNoiseFactoryFFT(psd, grid).make_random(seed=rng)())
            span = np.ptp(surface[aperture])
            if not np.isfinite(span) or span <= 0:
                raise ValueError("Generated surface has no finite peak-to-valley range.")
            surface = surface * (peaks[index] / span)
            # Subtract piston only inside the recorded mask, then enforce exact excluded zeros.
            surface = (surface - np.mean(surface[active])) * mask
            values[index] = surface
        yield values


def convert_commands(source, mask, samples, sets, repair_mask, provenance):
    """Convert either legacy CSV layout once, retaining source hashes and explicit repairs."""
    provenance["source_files"] = []
    for index in range(sets):
        path = Path(source) / f"modeval_dataset_{index}.csv"
        values = np.loadtxt(path, delimiter=",")
        if values.size != samples * 2500:
            raise ValueError(f"{path}: expected {samples * 2500} coefficients, got {values.size}.")
        values = values.reshape(samples, 2500)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite coefficients.")
        if repair_mask:
            values = values * mask
        provenance["source_files"].append({"path": str(path.resolve()), "sha256": checksum(path)})
        yield values


def parser():
    """Describe the offline generator and legacy-bank migration commands."""
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="action", required=True)
    for action in ["generate", "convert"]:
        sub = subcommands.add_parser(action)
        sub.add_argument("--output", type=Path, required=True, help="New bank directory; existing paths are rejected.")
        sub.add_argument("--mask", type=Path, required=True, help="Recorded binary 50x50 mask (.npy, .bin or CSV).")
        sub.add_argument("--samples", type=int, required=True)
        sub.add_argument("--sets", type=int, required=True)
        sub.add_argument("--units", choices=["um_surface"], required=True, help="Explicit units of the DM channel; no conversion.")
        if action == "generate":
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--ptv-min", type=float, default=0.01, help="Minimum surface PTV in micrometers.")
            sub.add_argument("--ptv-max", type=float, default=10**0.5, help="Maximum surface PTV in micrometers.")
            sub.add_argument("--exponent-min", type=float, default=-2.5)
            sub.add_argument("--exponent-max", type=float, default=-0.5)
        else:
            sub.add_argument("--source", type=Path, required=True)
            sub.add_argument("--repair-mask", action="store_true", help="Explicitly zero excluded positions in a new bank.")
    return result


def main(argv=None):
    """Prepare one new bank; parsing and generation never import hardware interfaces."""
    args = parser().parse_args(argv)
    if args.samples <= 0 or args.sets <= 0:
        raise ValueError("Samples and sets must be positive.")
    mask = load_mask(args.mask)
    provenance = {
        "action": args.action,
        "units": args.units,
        "mask_source": str(args.mask.resolve()),
        "mask_source_sha256": checksum(args.mask),
        "samples": args.samples,
        "sets": args.sets,
    }
    if args.action == "generate":
        import hcipy

        if args.seed < 0 or not all(np.isfinite(v) for v in [
            args.ptv_min, args.ptv_max, args.exponent_min, args.exponent_max
        ]) or not 0 < args.ptv_min <= args.ptv_max or args.exponent_min > args.exponent_max:
            raise ValueError("Invalid seed, surface PTV range, or spectrum exponent range.")
        provenance.update(
            seed=args.seed,
            hcipy_version=hcipy.__version__,
            ptv_min=args.ptv_min,
            ptv_max=args.ptv_max,
            exponent_min=args.exponent_min,
            exponent_max=args.exponent_max,
            piston="subtract mean over recorded active mask; excluded coefficients remain zero",
        )
        commands = random_commands(mask, args.samples, args.sets, args.seed, args.ptv_min, args.ptv_max,
                                   args.exponent_min, args.exponent_max)
    else:
        provenance["repair_mask"] = args.repair_mask
        commands = convert_commands(args.source, mask, args.samples, args.sets, args.repair_mask, provenance)
    write_bank(args.output, mask, args.samples, args.sets, commands, provenance)
    print(f"Prepared {args.sets * args.samples} commands in {args.output}")


if __name__ == "__main__":
    main()

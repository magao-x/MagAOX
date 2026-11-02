## \file artifacts.py
# \brief Verify complete worker bundles without importing torch or GPU libraries.
"""Eligibility is tied to the supervisor reset generation, not file existence."""

import json
from pathlib import Path


def engine_id(path):
    """Compute the C++ FNV-1a policy identity without importing torch/numpy."""
    value = 14695981039346656037
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            for byte in chunk:
                value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def verify_candidate(directory, root, generation):
    """Require matching engine/metadata/session identity before replacing the runtime pointer."""
    directory = Path(directory).resolve(strict=True)
    directory.relative_to(Path(root).resolve())
    engine = directory / "policy.trt"
    metadata = json.loads((directory / "policy.trt.json").read_text())
    if (
        not engine.stat().st_size
        or metadata["generation"] != generation
        or metadata["artifact_type"] != "tensorrt"
    ):
        raise ValueError("Incomplete, stale or incompatible candidate")
    from .configuration import revision

    if metadata["config_revision"] != revision(metadata["config"]):
        raise ValueError("Candidate configuration revision does not match metadata")
    model_id = engine_id(engine)
    if (
        metadata["active_model_id"] != model_id
        or not metadata.get("config_revision")
        or not metadata.get("session_id")
    ):
        raise ValueError("Candidate metadata does not match its engine/session")
    import numpy as np

    try:
        snapshot_file = np.load(
            directory / "policy.trt.training.npz", allow_pickle=False
        )
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(
            "Candidate training snapshot is missing or unreadable"
        ) from error
    if not isinstance(snapshot_file, np.lib.npyio.NpzFile):
        raise ValueError("Candidate training snapshot must be an NPZ archive")
    with snapshot_file as snapshot:
        saved = json.loads(str(snapshot["metadata_json"]))
        for key in (
            "active_model_id",
            "generation",
            "session_id",
            "config_revision",
            "config",
        ):
            if saved.get(key) != metadata[key]:
                raise ValueError(
                    "Candidate training snapshot identity mismatch: " + key
                )
        if saved.get("model_id") != model_id:
            raise ValueError("Candidate training snapshot model hash mismatch")
        policy = [name for name in snapshot.files if name.startswith("policy_state__")]
        dynamics = [name for name in snapshot.files if name.startswith("dynamics_")]
        if (
            not policy
            or not dynamics
            or any(not np.isfinite(snapshot[name]).all() for name in policy + dynamics)
        ):
            raise ValueError("Candidate training snapshot lacks finite model state")
    return directory, model_id

## \file library.py
# \brief Immutable saved policy/dynamics bundles, independent of live session eligibility.
"""Catalog names are stable IDs; only explicit loading admits saved state to a new session."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

from .artifacts import verify_candidate
from .process import atomic_json

FILES = ("policy.trt", "policy.trt.json", "policy.trt.training.npz")


def digest(path):
    """Hash a saved file without importing model/GPU libraries."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def compatible(saved, current):
    """Require identical model geometry; operational rates/penalties may change."""
    for group, fields in {
        "runtime": ("stream_n_modes", "n_modes_controlled"),
        "MDP": ("n_history", "control_delay"),
        "NN_models": ("policy_architecture",),
        "nonlinear": tuple(current["nonlinear"]),
    }.items():
        for field in fields:
            if saved[group][field] != current[group][field]:
                raise ValueError(f"Saved policy requires matching {group}.{field}")


class PolicyLibrary:
    """Own immutable directories beneath one shared, operator-provisioned root."""

    def __init__(self, root):
        """Create the library beneath the configured shared artifact location."""
        self.root = Path(root)
        if not self.root.is_absolute():
            raise ValueError("saved_engines must be an absolute directory")
        self.root.mkdir(mode=0o2775, parents=False, exist_ok=True)

    def catalog(self):
        """Read completed manifests only; hashing/model checks belong to explicit load."""
        entries = {}
        for path in sorted(self.root.glob("*/manifest.json")):
            if path.parent.name.startswith("."):
                continue
            try:
                manifest = json.loads(path.read_text())
                if (
                    manifest["format_version"] != 1
                    or manifest["id"] != path.parent.name
                    or not all(
                        isinstance(manifest[key], str)
                        for key in ("label", "created_utc", "model_id", "session_id")
                    )
                    or not isinstance(manifest["iteration"], int)
                    or manifest["iteration"] < 0
                    or not all((path.parent / name).is_file() for name in FILES)
                ):
                    continue
                entries[manifest["id"]] = manifest
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return entries

    def save(self, source, artifact_root, generation, label):
        """Copy one pinned acknowledged publication, then atomically expose its complete bundle."""
        source, model_id = verify_candidate(source, artifact_root, generation)
        label = label.strip() or "policy"
        if len(label) > 80 or any(ord(char) < 32 for char in label):
            raise ValueError("Use a label of at most 80 printable characters")
        stamp = datetime.now(timezone.utc)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "policy"
        identity = f"{slug}__{stamp:%Y%m%dT%H%M%SZ}__{uuid.uuid4().hex[:8]}"
        temporary = self.root / ("." + identity)
        target = self.root / identity
        temporary.mkdir()
        try:
            for name in FILES:
                shutil.copyfile(source / name, temporary / name)
            metadata = json.loads((temporary / "policy.trt.json").read_text())
            manifest = dict(
                format_version=1,
                id=identity,
                label=label,
                created_utc=stamp.isoformat(),
                model_id=model_id,
                iteration=metadata.get("iteration", 0),
                session_id=metadata["session_id"],
                source_revision=metadata.get("source_revision", "unrecorded"),
                calibration=metadata.get(
                    "calibration", "unknown; operator must verify"
                ),
                hashes={name: digest(temporary / name) for name in FILES},
            )
            atomic_json(temporary / "manifest.json", manifest)
            self.load_directory(temporary)
            temporary.rename(target)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load_directory(self, directory, config=None):
        """Verify hashes, coherent finite model state, and optional target geometry."""
        directory = Path(directory).resolve(strict=True)
        directory.relative_to(self.root.resolve())
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["format_version"] != 1 or set(manifest["hashes"]) != set(FILES):
            raise ValueError("Unsupported or incomplete saved bundle")
        for name in FILES:
            if digest(directory / name) != manifest["hashes"][name]:
                raise ValueError("Saved bundle checksum mismatch: " + name)
        metadata = json.loads((directory / "policy.trt.json").read_text())
        _, model_id = verify_candidate(directory, self.root, metadata["generation"])
        if model_id != manifest["model_id"]:
            raise ValueError("Saved bundle model identity mismatch")
        if config is not None:
            compatible(metadata["config"], config)
        return directory, manifest

    def load(self, identity, config):
        """Resolve only a current catalog ID, never a client-provided pathname."""
        if identity not in self.catalog():
            raise ValueError("Saved policy is no longer in the catalog")
        return self.load_directory(self.root / identity, config)

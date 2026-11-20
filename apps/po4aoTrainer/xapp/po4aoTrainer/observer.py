## \file observer.py
# \brief Explicit bounded diagnostic snapshots for an observing notebook.
"""Record matched diagnostic inputs; these files are not resumable training checkpoints."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """One iteration's config, identity, history and inputs for the existing plots."""

    metadata: dict
    observations: np.ndarray
    actions: np.ndarray

    @property
    def age_seconds(self):
        """Report elapsed wall-clock age without implying live attachment to a worker."""
        return max(0.0, time.time() - self.metadata["saved_unix_ns"] / 1e9)


def save_diagnostics(session, directory, *, keep=3, history_limit=1000):
    """Atomically write a completed diagnostic iteration into a caller-owned run directory.

    Retain at most ``keep`` snapshots in this directory. Store only the bounded
    recent replay used by the plots, not warmup/historical replay or optimizer
    checkpoints. The caller supplies a separate directory for each session.
    """
    if keep < 1 or history_limit < 1:
        raise ValueError("Diagnostic retention limits must be positive")
    record = None if session.last_result is None else session.last_result["record"]
    if record is None:
        raise RuntimeError("No completed diagnostic iteration is available")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "session_id": session.session_id,
        "normalization": session.normalization_metadata,
        "saved_unix_ns": time.time_ns(),
        "iteration": session.iteration,
        "config_revision": session.config_revision,
        "config": session.config,
        "history": session.metrics_history[-history_limit:],
    }
    serialized = json.dumps(metadata, allow_nan=False, sort_keys=True)
    target = (
        directory / f"diagnostics-{session.session_id}-{session.iteration:010d}.npz"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            np.savez_compressed(
                output,
                metadata=serialized,
                observations=session.obs_replay,
                actions=session.act_replay,
            )
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    for obsolete in sorted(directory.glob(f"diagnostics-{session.session_id}-*.npz"))[
        :-keep
    ]:
        obsolete.unlink()
    return target


def load_diagnostics(path):
    """Read a self-contained completed snapshot without importing models or connecting streams."""
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported diagnostic snapshot version")
        return DiagnosticSnapshot(
            metadata, stored["observations"].copy(), stored["actions"].copy()
        )

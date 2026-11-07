r"""\file sim_shmim.py
\brief Optional create-only ImageStreamIO boundaries for the existing local simulation; never attach DM devices.
"""
import os
from pathlib import Path
import socket
import uuid

import numpy as np


class SimImage:
    """A privately created image with distinct writer/reader views and coherent-counter checks."""
    def __init__(self, name, count, dtype, backend, image_class):
        """Create only a fresh simulation-prefixed image; refuse an existing path or incompatible layout."""
        if not name.startswith("ddspc_sim_"):
            raise ValueError("Only private simulation image names are permitted")
        self.path = Path(image_class._milk_shm_prefix) / (name + ".im.shm")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.count, self.dtype = count, np.dtype(dtype)
        self.owner = backend.Image()
        self.writer = self.reader = None
        self.inode = None
        try:
            result = self.owner.create(name, np.zeros((count, 1), dtype=self.dtype))
            if result not in (0, None):
                raise RuntimeError("Cannot create simulation image " + name)
            self.inode = self.path.stat().st_ino
            if list(self.owner.md.size[:2]) != [count, 1]:
                raise RuntimeError("Unexpected ImageStreamIO geometry; stop before wiring any application")
            self.writer, self.reader = image_class(name), image_class(name)
            self.last_read = np.zeros((count, 1), dtype=self.dtype)
            self.get_data(wait=False)
        except Exception:
            self.close()
            raise

    def counter(self):
        """Return a checked publication identity without reconnecting or silently accepting replacement."""
        if self.path.stat().st_ino != self.inode:
            raise RuntimeError("Simulation image was replaced")
        if self.reader.md.write != 0:
            raise RuntimeError("Simulation image publication is incomplete")
        return int(self.reader.md.cnt0)

    def write(self, value):
        """Publish a finite payload of the original dtype; never reinterpret ADC words as floats."""
        value = np.asarray(value)
        if value.size != self.count or value.dtype != self.dtype or not np.isfinite(value).all():
            raise ValueError("Simulation image payload shape/type mismatch")
        self.counter()
        self.writer.write(value.reshape(self.count, 1))

    def get_data(self, wait=False):
        """Copy once through magaox.Image; reject a concurrent publication or unexpected orientation/type."""
        if wait:
            raise ValueError("The deterministic simulation never waits on a consumer")
        before = self.counter()
        value = np.asarray(self.reader.get_data(wait=False))
        if before != self.counter() or value.size != self.count or value.dtype != self.dtype:
            raise RuntimeError("Incoherent or incompatible simulation image read")
        self.last_read = value.reshape(self.count, 1).copy()
        return self.last_read.copy()

    def close(self):
        """Close views and destroy only this run's still-identical owned image; leave replacements untouched."""
        for view in (self.reader, self.writer):
            if view is not None:
                view.close()
        self.reader = self.writer = None
        if self.inode is not None and self.path.exists() and self.path.stat().st_ino == self.inode:
            self.owner.destroy()
        self.inode = None


class SimImages:
    """Small optional transport adapter; the source, plant and controller algorithms remain unchanged."""
    def __init__(self, backend=None, image_class=None):
        """Allocate a unique create-only namespace using the existing ImageStreamIO/magaox image infrastructure."""
        if socket.gethostname().lower().startswith("exao"):
            raise RuntimeError("Development workstation only")
        if backend is None:
            try:
                import ImageStreamIOWrap as backend
                from magaox.shmim import Image as image_class
            except ImportError as error:
                raise RuntimeError("Use an isolated MagAO-X development runtime; no dependencies are installed here") from error
        self.prefix = f"ddspc_sim_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.images = {}
        try:
            for role, count, dtype in (("adc01", 2, np.uint16), ("pointing", 2, np.float32),
                                       ("source_tag", 8, np.uint64), ("disturbance", 2, np.float32),
                                       ("correction", 2, np.float32), ("wfs", 2, np.float32),
                                       ("consumed", 2, np.float32), ("state", 2, np.float32),
                                       ("frame_tag", 8, np.uint64)):
                self.images[role] = SimImage(self.prefix + "_" + role, count, dtype, backend, image_class)
        except Exception:
            self.close()
            raise

    def source_frame(self, raw, pointing, frame, first):
        """Publish/read four same-source ADC/pointing packets, committing explicit identity/time/counters last."""
        adc, position = np.empty((4, 2), np.uint16), np.empty((4, 2), np.float32)
        a, p, tag = (self.images[k] for k in ("adc01", "pointing", "source_tag"))
        for j in range(4):
            a.write(raw[j])
            p.write(pointing[j].astype(np.float32))
            metadata = np.array([1, frame * 4 + j, first + j * 125000, first + j * 125000 + 1000,
                                 frame, a.counter(), p.counter(), 0], np.uint64)
            tag.write(metadata)
            before = tag.counter()
            received = tag.get_data().ravel()
            adc[j] = a.get_data().ravel()
            position[j] = p.get_data().ravel()
            if before != tag.counter() or not np.array_equal(received, metadata) or received[5] != a.counter() or received[6] != p.counter():
                raise RuntimeError("Mismatched source packets")
        return adc, position[-1]

    def consumed(self, value, state, frame, now):
        """Publish and read actual simulator consumption, independently of its requested correction image."""
        u, x, tag = (self.images[k] for k in ("consumed", "state", "frame_tag"))
        u.write(value)
        x.write(state.astype(np.float32))
        metadata = np.array([1, frame, now, u.counter(), x.counter(), self.images["wfs"].counter(),
                             frame * 4 + 3, 2], np.uint64)
        tag.write(metadata)
        before = tag.counter()
        received, command = tag.get_data().ravel(), u.get_data().ravel()
        if before != tag.counter() or not np.array_equal(received, metadata) or received[3] != u.counter():
            raise RuntimeError("Mismatched consumed-input certificate")
        return command.copy()

    def close(self):
        """Release only this adapter's private simulation namespace."""
        for value in self.images.values():
            value.close()
        self.images.clear()

## \file shmim.py
# \brief Read and write named ImageStreamIO streams across producer restarts.
"""Shared-memory images in NumPy row/column order."""

import logging
import os
import pathlib
import threading
import time
from contextlib import contextmanager
from typing import NamedTuple, Optional

import ImageStreamIOWrap
import numpy as np

from .constants import DEFAULT_SHMIM_TIMEOUT_SEC

log = logging.getLogger(__name__)


class ShmimTimeout(Exception):
    """The reader's deadline expired before a fresh frame arrived."""


class ShapeMismatchException(Exception):
    """The image dimensions do not match the expected shape."""


class ShmimNotReady(RuntimeError):
    """The named stream is being created or replaced."""


class _Attachment(NamedTuple):
    """Identity and native dimensions captured on one successful open."""

    inode: int
    shape: tuple


@contextmanager
def _lock_until(lock, deadline, name):
    """Include contention with another reader or explicit reconnect in the read budget."""
    # A finite read can time out even when another thread is waiting indefinitely.
    timeout = -1 if deadline is None else max(0, deadline - time.monotonic())
    if not lock.acquire(timeout=timeout):
        raise ShmimTimeout(f"Timed out waiting for another reader of {name!r}")

    # Release ownership on both a successful read and any recovery error.
    try:
        yield
    finally:
        lock.release()


class Image(ImageStreamIOWrap.Image):
    """An ImageStreamIO handle that follows its name across file replacement."""

    _milk_shm_prefix = pathlib.Path('/milk/shm')
    _wait_slice_sec = 0.1

    def __init__(self, name):
        # Keep the name and directory available even while the mapping is closed.
        super().__init__()
        directory = pathlib.Path(os.environ.get('MILK_SHM_DIR') or self._milk_shm_prefix)
        if not directory.is_dir():
            directory = self._milk_shm_prefix if self._milk_shm_prefix.is_dir() else pathlib.Path('/tmp')

        # Serialize wrapper operations that use or replace this native mapping.
        self._name = name
        self._directory = directory
        self._path = directory / f"{name}.im.shm"
        self._lock = threading.RLock()
        self._attachment = None
        self.semID = None
        self._ensure_current()

    @property
    def name(self):
        """The stream name supplied by the caller."""
        return self._name

    @property
    def path(self):
        """The pathname followed when the producer replaces the stream."""
        return self._path

    @property
    def md(self):
        """Native metadata for an attached image, valid until the next close or reopen."""
        if self._attachment is None:
            raise RuntimeError(f"Stream {self.name!r} is detached")
        return super().md

    @property
    def shape(self):
        """Current dimensions in ImageStreamIO axis order."""
        with self._lock:
            return self._ensure_current().shape

    @property
    def counter(self):
        """Current frame counter from the named stream."""
        with self._lock:
            self._ensure_current()
            return self.md.cnt0

    def close(self):
        """Detach once and discard the reader slot associated with this mapping."""
        with self._lock:
            if self._attachment is None:
                return 0

            # Native close unmaps the image but leaves its C pointers populated.
            self.semID = None
            self._attachment = None
            return super().close()

    def open(self, name):
        """Open a named mapping and return the native status or INODE (48) if it was replaced."""
        with self._lock:
            # Update the requested path before attaching so a failed open can be retried.
            status = self.close()
            if status != 0:
                raise RuntimeError(f"ImageStreamIO could not close {self.name!r} (code {status})")
            self._name = name
            self._path = self._directory / f"{name}.im.shm"
            status = super().open(name)
            if status != 0:
                return status

            # The native binding retains dangling pointers after close and some failed opens.
            try:
                metadata = super().md
                if self.path.stat().st_ino != metadata.inode:
                    return 48
                self._attachment = _Attachment(metadata.inode, tuple(metadata.size))
                return 0
            except FileNotFoundError:
                return 40
            finally:
                if self._attachment is None:
                    super().close()

    def _ensure_current(self):
        """Open and verify the named file, or leave the handle ready for a retry."""
        # Keep the current mapping while the pathname is temporarily absent.
        inode = self.path.stat().st_ino
        previous = self._attachment
        if previous is not None and previous.inode == inode:
            return self._attachment

        # ImageStreamIO returns FILEOPEN (40) for both ENOENT and an incomplete file.
        status = self.open(self.name)
        if status == 40:
            with self.path.open('r+b'):
                raise ShmimNotReady(f"ImageStreamIO could not yet open {self.path}")
        if status == 48:
            raise ShmimNotReady(f"{self.path} changed during open")
        if status != 0:
            raise RuntimeError(f"ImageStreamIO could not open {self.name!r} (code {status})")

        # Report attachment changes once, independently of wait timeouts.
        if previous is not None:
            log.info("Reopened %s (inode %s -> %s, shape %s)", self.path, previous.inode, self._attachment.inode, self._attachment.shape)
        return self._attachment

    def _copy_pixels(self):
        """Copy pixels from the selected mapping in Python image order."""
        return np.squeeze(super().copy().T)

    def _snapshot(self):
        """Take an immediate snapshot and retain its native geometry across squeezing."""
        with self._lock:
            attachment = self._ensure_current()
            pixels = self._copy_pixels()
            if self.path.stat().st_ino != attachment.inode:
                raise ShmimNotReady(f"{self.path} changed during copy")
            return pixels, attachment

    def copy(self):
        """Take an immediate snapshot of the current stream in Python image order."""
        pixels, _ = self._snapshot()
        return pixels

    def _prepare_wait(self):
        """Claim a valid reader slot and discard posts predating this wait."""
        # A negative slot would make semflush flush every reader's semaphore.
        if self.semID is None:
            if self.md.sem == 0:
                self.close()
                raise ShmimNotReady(f"No initialized semaphores on {self.name!r}")
            slot = self.getsemwaitindex(0)
            if slot < 0:
                raise RuntimeError(f"No available reader semaphore for {self.name!r}")
            self.semID = slot

        # Flush once per fresh read or attachment, including after a producer restart.
        self.semflush(self.semID)

    def _wait_interval(self, deadline):
        """Limit the next wait or reopen delay to the remaining acquisition budget."""
        if deadline is None:
            return self._wait_slice_sec
        return min(self._wait_slice_sec, max(0, deadline - time.monotonic()))

    def _get_frame(self, deadline, check_before_wait, *, check_counter=True):
        """Wait for fresh pixels and return the attachment that supplied them."""
        # Use one budget for opening the file, waiting, and recovering from replacement.
        with _lock_until(self._lock, deadline, self.name):
            initial = self._attachment
            prepared = None
            check = check_before_wait or deadline is None or self.semID is None

            while True:
                # At the deadline make one final nonblocking check for an already published replacement.
                last_attempt = deadline is not None and time.monotonic() >= deadline

                # Check the path on entry when requested and after every unsuccessful slice.
                try:
                    attachment = self._ensure_current() if check else self._attachment
                    check = True
                    published = False
                    if prepared is not attachment:
                        baseline = self.md.cnt0 if attachment is initial else 0
                        self._prepare_wait()
                        prepared = attachment
                        published = self.md.cnt0 != baseline and not self.md.write

                    # A replacement may already have published its only frame while we were detached.
                    if not published:
                        status = self.semtimedwait(self.semID, self._wait_interval(deadline))
                        if status > 0:
                            raise RuntimeError(f"Invalid semaphore {self.semID} for {self.name!r}")
                        published = status == 0 and (not check_counter or self.md.cnt0 != baseline)

                    # Reject a copy if the pathname changed between notification and copying.
                    if published:
                        pixels = self._copy_pixels()
                        if self.path.stat().st_ino == attachment.inode:
                            return pixels, attachment

                except (FileNotFoundError, ShmimNotReady):
                    # Allow the producer to finish replacing the file without spinning.
                    time.sleep(self._wait_interval(deadline))

                # A quiet stream keeps its slot and all posts since the initial flush.
                if last_attempt:
                    raise ShmimTimeout(f"Timed out waiting for a fresh frame from {self.name!r}")

    def get_data(self, wait: bool = True, timeout_sec: Optional[float] = DEFAULT_SHMIM_TIMEOUT_SEC, check_before_wait: bool = False):
        """Copy current pixels, optionally waiting for a fresh frame.

        A finite timeout covers the wait and any reconnection. None waits until
        a fresh frame is available. check_before_wait also verifies the pathname
        before the first wait. All waiting reads check again during recovery.
        """
        # Immediate snapshots do not need a reader semaphore.
        if not wait:
            return self.copy()

        # Generic streams can signal updates with just a semaphore post.
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        pixels, _ = self._get_frame(deadline, check_before_wait, check_counter=False)
        return pixels

    def write(self, buffer: np.ndarray):
        """Write to the current stream in the column-major order expected by ImageStreamIO."""
        with self._lock:
            self._ensure_current()
            data = buffer if buffer.flags.f_contiguous else buffer.copy('F')
            return super().write(data)

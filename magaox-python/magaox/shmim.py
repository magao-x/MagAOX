from typing import Optional
import numpy as np
import ImageStreamIOWrap
import pathlib
import logging
from .constants import DEFAULT_SHMIM_TIMEOUT_SEC

log = logging.getLogger(__name__)

class ShmimTimeout(Exception):
    pass

class ShapeMismatchException(Exception):
    pass

class Image(ImageStreamIOWrap.Image):
    '''Customized Image from ImageStreamIOWrap that enforces transposition
    (i.e. C-ordering) at the copy() and write() boundary for consistency
    with Python image plotting
    '''
    _milk_shm_prefix = pathlib.Path('/milk/shm')
    _opened_with_inode : int
    semID : Optional[int] = None
    def _get_path(self, name):
        return self._milk_shm_prefix / f"{name}.im.shm"

    @property
    def name(self):
        return self.md.name

    @property
    def path(self) -> pathlib.Path:
        return self._get_path(self.md.name)

    def __init__(self, name):
        super().__init__()
        self._reopen(name)

    def _check_inode(self):
        return self.path.stat().st_ino == self._opened_with_inode

    def _reopen(self, name):
        this_path = self._get_path(name)
        if not this_path.exists():
            raise FileNotFoundError(f"Looked for {name} at {self._get_path(name)} but no such file exists")
        try:
            self.close()
        except RuntimeError:
            # raised because Image wasn't open yet, ignore
            pass
        ret = self.open(name)
        if ret != 0:
            raise RuntimeError(f"ImageStreamIO could not open {repr(name)}")
        assert self.md.name == name
        self._opened_with_inode = self.md.inode
        assert this_path.stat().st_ino == self._opened_with_inode

    def copy(self):
        if not self._check_inode():
            log.info(f"Reopening {self.path} because we detected an inode change")
            self._reopen(self.name)
        return np.squeeze(super().copy().T)

    def get_data(self, wait: bool=True, timeout_sec: Optional[float]=DEFAULT_SHMIM_TIMEOUT_SEC, check_before_wait: bool=False):
        '''Get a copy of the image data, optionally waiting (with an
        optional timeout) for an updated frame to arrive before
        returning

        We compare the inode of the shmim file to the one we got when we opened it
        before attempting to await it. When no timeout is set, we always check to ensure
        we don't try to read stale data. When a timeout is set, we only check after timing out
        unless `check_before_reading` is `True`. When `wait == False` we check and reopen
        the shmim before copying.

        Parameters
        ----------
        wait : bool (default: True)
            Whether to block until the underlying shmim has an update
        timeout_sec : float (default: 5.0)
            Number of seconds to wait before giving up (pass `None` to wait forever)
        check_before_wait : bool (default: False)
            Check the inode of the shmim before awaiting its semaphore (always True when no timeout supplied)
        '''
        log.debug(f'get_data({wait=}, {timeout_sec=}, {check_before_wait=})')
        if self.semID is None:
            self.semID = self.getsemwaitindex(0)
        if wait:
            # ensure we're caught up by zeroing the semaphore before waiting
            log.debug("ensure we're caught up by zeroing the semaphore before waiting")
            self.semflush(self.semID)
            if timeout_sec is None:
                # We need to detect size changes before we wait or we could wait forever
                # on a stale shmim
                log.debug('We must check before waiting because otherwise we may already have a stale shmim and wait forever')
                if not self._check_inode():
                    log.debug('Reopening bc check before wait found inode change')
                    self._reopen(self.name)
                # Wait until another process writes to this shmim
                log.debug('waiting on semaphore, no timeout')
                self.semwait(self.semID)
            else:
                log.debug(f"{timeout_sec=}")
                if check_before_wait:
                    log.debug('check_before_wait=True')
                    if not self._check_inode():
                        log.info(f"Reopening {self.path} because we detected an inode change")
                        log.debug(f"Reopening {self.path} because we detected an inode change")
                        self._reopen(self.name)
                else:
                    log.debug('not checking before waiting')
                ret = self.semtimedwait(self.semID, timeout_sec)
                log.debug(f"{ret=} (first)")
                if ret != 0:
                    log.debug('first wait failed')
                    if not self._check_inode():
                        log.debug('post first wait reopening')
                        self._reopen(self.name)
                        log.debug('post first wait reopening done')
                    log.info(f"Reopening {self.path} because we timed out and detected an inode change")
                    ret = self.semtimedwait(self.semID, timeout_sec)
                    log.debug(f"{ret=} (recovery)")
                if ret != 0:
                    raise ShmimTimeout(f"Timed out after {2 * timeout_sec} sec without new data")
        else:
            log.debug('not waiting')
            if not self._check_inode():
                log.debug(f"not waiting, but checking inode shows change so reopen")
                self._reopen(self.name)
                log.info(f"Reopening {self.path} because we detected an inode change")
        log.debug(f"Returning a copy of {self.name}")
        return self.copy()

    def write(self, buffer: np.ndarray):
        '''Set the contents of the shmim, reordering buffer to
        column-major if necessary
        '''
        if not self._check_inode():
            self._reopen(self.name)
            log.info(f"Reopening {self.path} because we detected an inode change")
        if not buffer.flags['F_CONTIGUOUS']:
            data_towrite = buffer.copy('F')
        else:
            data_towrite = buffer
        return super().write(data_towrite)

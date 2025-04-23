from typing import Optional
import numpy as np
import ImageStreamIOWrap

class ShmimTimeout(Exception):
    pass

class ShapeMismatchException(Exception):
    pass

class Image(ImageStreamIOWrap.Image):
    '''Customized Image from ImageStreamIOWrap that enforces transposition
    (i.e. C-ordering) at the copy() and write() boundary for consistency
    with Python image plotting
    '''
    semID : Optional[int] = None
    def __init__(self, name):
        super().__init__()
        ret = self.open(name)
        if ret != 0:
            raise RuntimeError(f"Could not open {repr(name)}")

    def copy(self):
        return np.squeeze(super().copy().T)

    def get_data(self, check: bool=False, timeout: Optional[float]=None):
        '''Get a copy of the image data, optionally waiting (with an
        optional timeout) for an updated frame to arrive before
        returning

        Parameters
        ----------
        check : bool
            Whether to block until the underlying shmim has an update
        timeout : float
            Number of seconds to wait before giving up
        '''
        if check:
            if self.semID is None:
                self.semID = self.getsemwaitindex(0)
            
            self.semflush(self.semID)
            if timeout is None:
                self.semwait(self.semID)
            else:
                if timeout < 0:
                    raise ValueError(f"Invalid timeout: {timeout}")
                ret = self.semtimedwait(self.semID, timeout)
                if ret != 0:
                    raise ShmimTimeout(f"Timed out after {timeout} sec without new data")
        return self.copy()

    def write(self, buffer: np.ndarray):
        '''Set the contents of the shmim, reordering buffer to
        column-major if necessary
        '''
        if not buffer.flags['F_CONTIGUOUS']:
            data_towrite = buffer.copy('F')
        else:
            data_towrite = buffer
        return super().write(data_towrite)

from hcipy import *
import numpy as np
from scipy.linalg import hadamard
import time

from magpyx.utils import ImageStream


class XDeformableMirror(DeformableMirror):
    def __init__(self, dm='woofer', channel=2, mode_basis_generator=None):
        ''' The MagAO-X python interface for the deformable mirrors.

        Parameters
        ----------
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        mode_basis_generator : Field generator
            A function that calculates the modes for the DM. Default uses the full actuator basis.
        '''
        
        if dm == 'dmwoofer':
            self.dmindex = 0
            self.num_across = 11
            self.size = np.array([1, 1])
        elif dm == 'dmtweeter':
            self.dmindex = 1
            self.num_across = 50
            self.size = np.array([1, 1])
        elif dm == 'dmncpc':
            self.dmindex = 2
            self.num_across = 34
            self.size = np.array([1, np.sqrt(2.0)])
        else:
            self.dmindex = 0
            self.num_across = 11

        if mode_basis_generator is None:
            mode_basis_generator = lambda grid : ModeBasis(np.eye(grid.size), grid)
        
        # DM write channel
        self.dmwrite = ImageStream('dm0{:d}disp0{:d}'.format(self.dmindex, channel))
        
        # The DM modes
        self.grid = make_pupil_grid(self.num_across, 1)
        modes = mode_basis_generator(self.grid)
        super().__init__(modes)

    def send(self, sleep=None):
        self.dmwrite.write(self.surface.shaped.astype(np.float32))        
        if sleep is not None:
            time.sleep(sleep)
    
    def reset(self):
        self.flatten()
        self.send()

class XZernikeMirror(XDeformableMirror):
    def __init__(self, starting_mode=2, number_of_modes=5, dm='woofer', channel=2):
        ''' The MagAO-X python interface for a Zernike-modes basis deformable mirror.
        Parameters
        ----------
        starting_mode : int
            The index of the first mode of the mode basis.
        number_of_modes : int
            The number of modes in the created mode basis.
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        '''

        mode_basis_generator = lambda grid: make_zernike_basis(number_of_modes, 1.0, grid, starting_mode=starting_mode)
        super().__init__(dm=dm, channel=channel, mode_basis_generator=mode_basis_generator)

class XFourierMirror(XDeformableMirror):
    def __init__(self, fourier_grid, dm='woofer', channel=2):
        ''' The MagAO-X python interface for a Zernike-modes basis deformable mirror.
        Parameters
        ----------
        fourier_grid : grid
            The spatial frequencies of the each mode in units of lambda/D.
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        '''
        mode_basis_generator = lambda grid: make_fourier_basis(grid, fourier_grid.scaled(2 * np.pi))
        super().__init__(dm=dm, channel=channel, mode_basis_generator=mode_basis_generator)

class XPokeMirror(XDeformableMirror):
    def __init__(self, poke_grid, dm='woofer', channel=2):
        ''' The MagAO-X python interface for a Zernike-modes basis deformable mirror.
        Parameters
        ----------
        poke_grid : grid
            The position of the actuator pokes.
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        '''
        if dm == 'woofer':
            actuator_spacing = 1.0
        elif dm == 'tweeter':
            actuator_spacing = 1.0
        elif dm == 'ncpc':
            actuator_spacing = 1.0
        else:
            actuator_spacing = 1.0

        crosstalk = 0.15 # Typical value for BMCs.
        sigma = actuator_spacing / (np.sqrt((-2 * np.log(crosstalk))))
        mode_basis_generator = lambda grid: make_gaussian_pokes(grid, poke_grid, sigma, cutoff=5)
        super().__init__(dm=dm, channel=channel, mode_basis_generator=mode_basis_generator)


class XHadamardMirror(XDeformableMirror):
    def __init__(self, dm='woofer', channel=2):
        ''' The MagAO-X python interface for a Zernike-modes basis deformable mirror.
        Parameters
        ----------
        dm : string
            The name of the dm that we are connecting to.
        channel : int
            The dm channel that is being used.
        '''
        
        from scipy.linalg import hadamard
        def make_hadamard_modes(grid):
            grid.shape

        #super().__init__(dm=dm, channel=channel, mode_basis_generator)
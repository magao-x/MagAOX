import hcipy as hp
import numpy as np
import time
import datetime
import os

### Make the save directory
def make_savedir_for_today(start_path='./Data/'):
    ct = datetime.datetime.now()
    savedir = start_path + '{:>04d}{:>02d}{:>02d}/'.format(ct.year, ct.month, ct.day)
    make_savedir(savedir)
    return savedir

def make_savedir(new_path):
    try:
        os.makedirs(new_path, exist_ok=True)
    except FileExistsError:
        print("Directory already exists.")

def make_time_stamp():
    t = time.time()
    t_seconds = int(t)
    ms = int(1000 * (t - t_seconds))
    stamp = '{:d}{:d}'.format(t_seconds, ms)
    return stamp

def wait_for_ready(client, device, timeout=0.1):
    time.sleep(timeout)
    while client[device + '.fsm.state'] != 'READY':
        time.sleep(timeout)

def make_horizontal_probe(grid, direction='left'):
    probe = grid.zeros()
    probe = probe.shaped

    width = 4
    height = 4
    x = (-1.0)**(np.arange(width) - width//2)
    y = (-1.0)**(np.arange(height) - height//2)

    # 34 - 15 16 17 18 19 20
    probe[15:19, 6:10] = np.outer(x, y)
    probe[15:19, 6:10] = np.outer(y, x)
    if direction == 'right':
        probe = probe[:,::-1]

    return probe.ravel()

def make_vertical_probe(grid, direction='down'):
    probe = grid.zeros()
    probe = probe.shaped

    width = 4
    height = 4
    x = (-1.0)**(np.arange(width) - width//2)
    y = (-1.0)**(np.arange(height) - height//2)

    # 34 - 15 16 17 18 19 20
    probe[8:12, 15:19] = np.outer(y, x)
    probe[8:12, 15:19] = np.outer(x, y)
    probe = np.sign(probe)
    if direction == 'up':
        probe = probe[::-1,:]

    return probe.ravel()

def make_ncpc_alignment_poke_pattern():
    ncpc_act_grid = hp.make_pupil_grid(34, 34/30.0 * np.array([1.0, np.sqrt(2)]))
    probe = make_vertical_probe(ncpc_act_grid) + make_vertical_probe(ncpc_act_grid, 'up')
    probe += make_horizontal_probe(ncpc_act_grid) + make_horizontal_probe(ncpc_act_grid, 'right')
    return probe


def wait_for_state(client, indi_property, value, wait_time=1, tolerance=None):
    if tolerance is None:
        while not (client[indi_property] == value):
            print("Waiting...")
            time.sleep(wait_time)
    else:
        while not abs(client[indi_property] - value) < tolerance:
            print("Waiting...")
            time.sleep(wait_time)

def calibrate_indi_device(client, device_name, device_property, delta_pertubation, camera, measurement_function, num_stack=50, do_wait=False):
    client.get_properties('{:s}'.format(device_name))

    property_current = '{:s}.{:s}.current'.format(device_name, device_property)
    property_target = '{:s}.{:s}.target'.format(device_name, device_property)
    current_position = client[property_current]
    print("Probe to {:f} +- {:f}".format(current_position, delta_pertubation))
    slope = 0
    for s in [-1, 1]:
        client[property_target] = current_position + s * delta_pertubation
        if do_wait:
            wait_for_state(client, property_current, current_position + s * delta_pertubation, tolerance=0.1 * delta_pertubation)
        else:
            time.sleep(5)

        # Take measurement
        camera.grab_stack(3)
        im = camera.grab_stack(num_stack)

        measurement = measurement_function(im)
        slope += s * measurement / (2 * delta_pertubation)

    client[property_target] = current_position
    if do_wait:
        wait_for_state(client, property_current, current_position, tolerance=0.1 * delta_pertubation)
    else:
        time.sleep(2)

    return slope

class XCorrShift():
    def __init__(self, reference_image, domain_pixels=480, domain_size=40, filter_size=None, reference_point=np.array([0,0])):
        self._reference_image = reference_image
        self._xgrid = hp.make_pupil_grid(domain_pixels, domain_size).shifted(reference_point)

        self._fft = hp.FastFourierTransform(self._reference_image.grid)
        self._mft = hp.MatrixFourierTransform(self._xgrid, self._fft.output_grid)

        self._filter_size = filter_size
        if filter_size is not None:
            # Change this to a super gaussian filter to remove ringing.
            self._spatial_filter = hp.make_circular_aperture(self._filter_size)(self._fft.output_grid)
        else:
            self._spatial_filter = 1

        self._kernel = np.conj(self._fft.forward(self._reference_image + 0j))

    def cross_correlate(self, image, local_reference_point=np.array([0,0])):
        tilt = np.exp(1j * local_reference_point @ self._fft.output_grid.points.T)
        xcorr = np.real(self._mft.backward(self._fft.forward(image + 0j) * self._spatial_filter * self._kernel * tilt))
        return xcorr

    def _subpixel_offset_from_quadratic(self, xcorr_2d, ix, iy):
        ny, nx = xcorr_2d.shape
        if ix < 1 or ix > nx - 2 or iy < 1 or iy > ny - 2:
            return None

        patch = xcorr_2d[iy-1:iy+2, ix-1:ix+2]
        dx, dy = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
        A = np.vstack([
            dx.ravel()**2,
            dy.ravel()**2,
            (dx * dy).ravel(),
            dx.ravel(),
            dy.ravel(),
            np.ones(9)
        ]).T
        z = patch.ravel()

        coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
        a, b, c, d, e, _ = coeff

        denom = 4*a*b - c*c
        if abs(denom) < 1e-12:
            return None

        x0 = (c*e - 2*b*d)/denom
        y0 = (c*d - 2*a*e)/denom

        if abs(x0) > 1.0 or abs(y0) > 1.0:
            return None

        return np.array([x0, y0])

    def _subpixel_peak_position(self, xcorr):
        xcorr_arr = np.asarray(xcorr)
        grid_shape = tuple(self._xgrid.shape)
        xcorr_2d = xcorr_arr.reshape(grid_shape)

        idx_max = np.argmax(xcorr_2d)
        iy, ix = np.unravel_index(idx_max, grid_shape)

        offset = self._subpixel_offset_from_quadratic(xcorr_2d, ix, iy)
        if offset is None:
            return self._xgrid.points[idx_max]

        dx, dy = offset
        x_center = self._xgrid.x.reshape(grid_shape)[iy, ix]
        y_center = self._xgrid.y.reshape(grid_shape)[iy, ix]

        return np.array([
            x_center + dx * self._xgrid.delta[0],
            y_center + dy * self._xgrid.delta[1]
        ])

    def measure(self, image, local_reference_point=np.array([0,0]), subpixel=True):
        xcorr = self.cross_correlate(image, local_reference_point)

        if subpixel:
            peak = self._subpixel_peak_position(xcorr)
        else:
            indx_max = np.argmax(np.asarray(xcorr))
            peak = self._xgrid.points[indx_max]

        return peak + local_reference_point
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
    def __init__(self, reference_image, domain_pixels=480, domain_size=40, filter_size=None):
        self._reference_image = reference_image
        self._xgrid = hp.make_pupil_grid(domain_pixels, domain_size)

        self._fft = hp.FastFourierTransform(self._reference_image.grid)
        self._mft = hp.MatrixFourierTransform(self._xgrid, self._fft.output_grid)

        self._filter_size = filter_size
        if filter_size is not None:
            # Change this to a super gaussian filter to remove ringing.
            self._spatial_filter = hp.make_circular_aperture(self._filter_size)(self._fft.output_grid)
        else:
            self._spatial_filter = 1

        self._kernel = np.conj(self._fft.forward(self._reference_image + 0j))

    def cross_correlate(self, image):
        xcorr = np.real(self._mft.backward(self._fft.forward(image + 0j) * self._spatial_filter * self._kernel))
        return xcorr

    def measure(self, image):
        # Do a cross-correlation and find the peak pixel
        # TODO: implement sub-pixel precision with polynomial fitting
        xcorr = self.cross_correlate(image)
        indx_max = np.argmax(xcorr)
        return self._xgrid.points[indx_max]
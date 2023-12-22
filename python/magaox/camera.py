from hcipy import *
import numpy as np

from pyMilk.interfacing.isio_shmlib import SHM as shm
import purepyindi2 as indi


class XCam():
	'''A python interface for existing MagAO-X cameras.
	# TODO: add a background thread that pulls the remote indi properties once a second.
	'''
	def __init__(self, shm_name, pixel_size=1, use_hcipy=False):
		self.shm_name = shm_name
		
		self._client = indi.client.IndiClient()
		self._client.connect()
		self._client.get_properties(shm_name)
		self._local_indi_prop = {}
		
		self._old_counter = 0
		self._need_reconnect = False
		self._use_hcipy = use_hcipy
		self._pixel_size = pixel_size

		self.connect_camera()	

	def connect_camera(self):
		self.shmim = shm(self.shm_name)
		
		# A dark frame might not exists. Check for existence!
		self.dark_shmim = shm(self.shm_name + '_dark')
		self._need_reconnect = False

		if self._use_hcipy:
			self.grid = make_pupil_grid(self.shape, self._pixel_size * self.shape)

	def pull_indi_properties(self):
		'''Pulls the values of all indi properties of the device from the origin.
		'''
		self._local_indi_prop = {}
		for prop in self._client[self.shm_name]:
			for sub_prop in self._client[self.shm_name][prop]:
				if prop not in self._local_indi_prop:
					self._local_indi_prop[prop] = {}
				self._local_indi_prop[prop][sub_prop] = self._client[self.shm_name][prop][sub_prop]

	def reconnect_on_change(self, properties=['roi_region_h', 'roi_region_w', 'roi_region_bin_x', 'roi_region_bin_y']):
		'''Compares the local indi properties against the remote values and reconnects the camera if it has changed.

		properties - array like
			The properties that require the camera to reconnect if changed. The default values are all the parameters that impact shmim size.
		'''		
		has_changed = False

		for prop in properties:
			if self._local_indi_prop[prop]['current'] != self._client['{:s}.{:s}.current'.format(self.shm_name, prop)]:
				has_changed = True
				break
		
		if has_changed:
			self.pull_indi_properties()
			self.connect_camera()

	@property
	def counter(self):
		return self.shmim_meta_data.cnt0

	@property
	def shmim_meta_data(self):
		return self.shmim.IMAGE.md

	@property
	def meta_data(self):
		meta_dict = {}
		for prop1 in self._local_indi_prop:
			for prop2, value in self._local_indi_prop[prop1].items():
				meta_dict['{:s}.{:s}'.format(prop1, prop2)] = value
		return meta_dict

	@property
	def shape(self):
		return np.array([self.shmim_meta_data.size[0], self.shmim_meta_data.size[1]])

	@property
	def exposure_time(self):
		return self._client[self.shm_name + '.exptime.current']
	
	@exposure_time.setter
	def exposure_time(self, new_exposure_time):
		# Check if within valid ranges?
		self._client[self.shm_name + '.exptime.target'] = new_exposure_time

	@property
	def emgain(self):
		if 'emgain' in self._client[self.shm_name]:
			return self._client[self.shm_name + '.exptime.current']
		else:
			raise MissingValueError("This camera has no emgain.")

	@emgain.setter
	def emgain(self, new_emgain):
		if 'emgain' in self._client[self.shm_name]:
			self._client[self.shm_name + '.emgain.target'] = new_emgain
		else:
			raise MissingValueError("This camera has no emgain.")

	@property
	def temperature(self):
		if 'temp_ccd' in self._client[self.shm_name]:
			return self._client[self.shm_name + '.temp_ccd.current']
		else:
			raise MissingValueError("This camera has no temperature monitor.")

	@temperature.setter
	def temperature(self, new_temperature):
		if 'temp_ccd' in self._client[self.shm_name]:
			self._client[self.shm_name + '.temp_ccd.target'] = new_temperature
		else:
			raise MissingValueError("This camera has no temperature monitor.")

	@property
	def shutter(self):
		# TODO: check if I have the True and False correct
		return True if self.properties['shutter']['toggle'] == indi.SwitchState.ON else False

	@shutter.setter
	def shutter(self, shutter_state):
		self._client[self.shm_name + '.shutter.toggle'] = indi.SwitchState.ON if shutter_state else indi.SwitchState.OFF

	def grab(self):
		self._old_counter = self.counter
		data = self.shmim.get_data(check=True, timeout=5 * self.exposure_time).astype(float)
		
		if self.counter == self._old_counter:
			self._need_reconnect = True
		else:
			self._old_counter = self.counter
		
		if np.all(self.dark_shmim.IMAGE.md.size==self.shmim.IMAGE.md.size):
			data -= self.dark_shmim.get_data(check=False).astype(float)

		if self._use_hcipy:
			data = Field(data.ravel(), self.grid)

		return data
			
	def grab_many(self, num_images):
		OUT = np.zeros((num_images, *self.shmim.shape), dtype=self.shmim.nptype)

		if np.all(self.dark_shmim.IMAGE.md.size==self.shmim.IMAGE.md.size):
			background = self.dark_shmim.get_data(check=False)

		for i in range(num_images):
			self._old_counter = self.counter

			OUT[i] = self.shmim.get_data(check=True, timeout=5 * self.exposure_time) - background
			
			if self.counter == self._old_counter:
				self._need_reconnect = True
				break
			else:
				self._old_counter = self.counter
		
		if self._use_hcipy:
			OUT = Field(OUT.reshape((num_images,-1)), self.grid)

		return OUT
	
	def grab_stack(self, num_images):
		stacked_image = 0
		k = 0
		for i in range(num_images):
			self._old_counter = self.counter
			
			image = self.shmim.get_data(check=True, timeout=5 * self.exposure_time)
			
			if self.counter == self._old_counter:
				self._need_reconnect = True
				stacked_image = 0
				k = 0
			else:
				self._old_counter = self.counter
				stacked_image += image
				k += 1
		
		if k != 0:
			stacked_image = stacked_image / k

		if np.all(self.dark_shmim.IMAGE.md.size==self.shmim.IMAGE.md.size):
			stacked_image -= self.dark_shmim.get_data(check=False)
		
		if self._use_hcipy:
			stacked_image = Field(stacked_image.ravel(), self.grid)

		return stacked_image
		
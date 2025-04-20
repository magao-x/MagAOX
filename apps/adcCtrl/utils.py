import numpy as np
import hcipy as hp
from scipy.optimize import minimize
import time


class AdcFitter:
    def __init__(self,wavelength=656E-9,bandwidth=100E-9,grating_angle=28,grating_freq=47):
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.grating_angle = grating_angle
        self.grating_freq = grating_freq
        self.normalized_wavelength = wavelength / 656E-9 #normalizing the wavelengths to the ha values
        self.normalized_bandwidth = bandwidth / 656E-9 
        self.maxiter = 6
        self.control_mtx = np.matrix([[0,0],])
        self.current_speckle = None

    '''this is all the gaussian fitting stuff'''
    def set_measurement(self, data):
        self.data = data

    def make_gaussian(self, mu_x, mu_y, sigma_x,sigma_y, orientation):
        def func(grid):
            new_grid = grid.shifted([-mu_x, -mu_y]).rotated(orientation)
            x = new_grid.x / sigma_x
            y = new_grid.y / sigma_y
            r2 = x**2 + y**2
            return hp.Field(np.exp(-0.5 * r2), grid)
        return func

    def satellite_spot(self, amplitude, mu_x, mu_y, sigma_x, sigma_y, orientation, background):
        def func(grid):
            return hp.Field(amplitude * self.make_gaussian(mu_x, mu_y, sigma_x, sigma_y, orientation)(grid) + background, grid) 
        return func

    def cost(self, theta):
        fit = self.satellite_spot(*theta)(self.data.grid)
        j = np.sum( (self.data - fit)**2)

        aspect_ratio = np.abs(theta[3] / theta[4])
        #print(f'current speckle: {self.current_speckle}')

        ##boundary condition for the aspect ratio
        if self.current_speckle == 0 or self.current_speckle == 2:
            if aspect_ratio  >= 1:
                j+= 1E3 * aspect_ratio **2
        elif self.current_speckle ==1 or self.current_speckle == 3:
            if aspect_ratio <= 1:
                j+= 1E3 * 1/aspect_ratio**2

        return j
    
    def fit(self,theta_est):
        fitting = minimize(self.cost,theta_est,options={'maxiter':self.maxiter})
        return fitting
    
    def estimate_centroid(self):
        M00 = np.sum(self.data)
        M10 = np.sum(self.data * self.data.grid.x)
        M01 = np.sum(self.data * self.data.grid.y)

        centroid = [M10/M00,M01/M00]
        return centroid
    
    def estimate_angle(self):
        M00 = np.sum(self.data)
        M10 = np.sum(self.data * self.data.grid.x)
        M01 = np.sum(self.data * self.data.grid.y)
        
        M20 = np.sum(self.data * self.data.grid.x**2)
        M02 = np.sum(self.data * self.data.grid.y**2)
        M11 = np.sum(self.data * self.data.grid.y * self.data.grid.x)

        mu10 = M10 / M00
        mu01 = M01 / M00
        mu20 = M20 / M00 - mu10**2
        mu02 = M02 / M00 - mu01**2
        mu11 = M11 / M00 - mu10 * mu01
        angle = (1/2 * np.arctan2(2 * mu11, mu20 - mu02))

        return angle

    def find_speckle(self, image,speckle_number):
        #this assumes that the units of the image x and y axes are lam/d at 656 nm. will this always be the case?
        '''speckles are indexed from the top right going counter clockwise'''
        grating_freq = self.grating_freq
        grating_angle = self.grating_angle
        corners = np.array([[0, grating_freq * self.normalized_wavelength],[grating_freq * self.normalized_wavelength,0],[0, -grating_freq * self.normalized_wavelength],[-grating_freq * self.normalized_wavelength,0]])
        sizes = np.array([[8,20],[20,8],[8,20],[20,8]])
        #sizes = np.array([[16,25],[25,16],[16,25],[25,16]])

        rect = hp.make_rotated_aperture(hp.make_rectangular_aperture(size=sizes[speckle_number], center=corners[speckle_number]), np.deg2rad(-grating_angle))(image.grid)
        speckle_img = rect * image
        return speckle_img
    

    def set_psf(self,psf):
        self.psf = psf  

    def find_speckle_angles2(self):

        speckle_angles = np.zeros(4)
        sig_x = [0.8,3.5,0.8,3.5]
        sig_y = [3.5,0.8,3.5,0.8]

        for i in range(4):
            img = self.find_speckle(self.psf,i)
            self.current_speckle = i
            self.set_measurement(img)

            sigma_x = sig_x[i]
            sigma_y = sig_y[i]
            #orientation = np.radians(28)

            orientation = self.estimate_angle()
            if self.current_speckle == 0 or self.current_speckle ==2:
                orientation = np.pi/2 - orientation
            elif self.current_speckle == 1 or self.current_speckle == 3:
                orientation = -orientation

            #params common to all four
            amplitude = self.data.max()
            centroid = self.estimate_centroid()
            mu_x = centroid[0]
            mu_y = centroid[1]
            background = 0

            theta_est = np.array([amplitude, mu_x, mu_y, sigma_x, sigma_y, orientation, background])
            fit = self.fit(theta_est) 
            speckle_angles[i] = np.degrees(fit.x[5])

        self.current_speckle = None
        speckle_angles = np.array(speckle_angles).T

        return speckle_angles
    
    def speckle_pairs(self, speckle_angles):
        pair02 = speckle_angles[0] - speckle_angles[2]
        pair13 = speckle_angles[1] - speckle_angles[3]
        return np.array([pair02,pair13])

    def calculate_command(self,speckle_angles):
        predicted_disp = self.control_mtx * np.matrix(speckle_angles).T
        predicted_disp = np.array(predicted_disp)
        return -predicted_disp

    def clear(self):
        self.psf = None
        self.data = None

    def set_control_mtx(self,matrix):
        self.control_mtx = matrix

    '''this is stuff specifically for working with the real calibration datacubes'''
    def window_field(self,data, center, width, height):
        indx = data.grid.closest_to(center)
        y_ind, x_ind = np.unravel_index(indx, data.shaped.shape)
        cutout = data.shaped[(y_ind-height//2):(y_ind + height//2), (x_ind-width//2):(x_ind+width//2)]
        sub_grid = hp.make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
        return hp.Field(cutout.ravel(), sub_grid)
    
    def crop_image(self, image,extent=400,mask_diam=60): 
        #cutout a centered PSF
        img = image/image.max()

        img_subtracted = img >0.1
        center_of_intensity = np.array([sum(img_subtracted*img_subtracted.grid.x)/sum(img_subtracted),sum(img_subtracted*img_subtracted.grid.y)/sum(img_subtracted)])
        mask_ap = hp.make_circular_aperture(mask_diam,center_of_intensity)
        mask = mask_ap(img_subtracted.grid)
        mask = abs(mask - 1)
        masked_img = mask * img

        img = masked_img
        img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],extent,extent)
        img /= img.max()
        
        bk = np.median(img)
        img -= bk
        img = hp.Field([x if x>0 else 0 for x in img],img.grid)

        return img

    def crop_cube(self, data_cube,extent=400,mask_diam=60): #if you add the wavelength in here it'll make sure that nothing gets cut off of the edges
        cropped_cube = []
        for i in range(len(data_cube)):
            img = self.crop_image(data_cube[i],extent,mask_diam)
            cropped_cube.append(img)
        return cropped_cube


    def filter_image(self,img,low_freq = 0.01,high_freq=1):

        ff = hp.FourierFilter(img.grid, hp.make_circular_aperture(2 * np.pi * low_freq))
        filtered_img= np.real(ff.forward(img + 0j))
        img = img - filtered_img

        ff2 = hp.FourierFilter(img.grid, hp.make_circular_aperture(2 * np.pi * high_freq))
        filtered_img = np.real(ff2.forward(img + 0j))
        img = filtered_img

        # binc, profile, std_profile, ncount = radial_profile(img,.25)
        # r_coordinates = img.grid.as_('polar').r
        # radial_map = np.interp(r_coordinates, binc, profile)

        filtered_subtracted = img #- radial_map
        
        return filtered_subtracted


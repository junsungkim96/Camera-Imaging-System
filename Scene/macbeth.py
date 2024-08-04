import numpy as np
from copy import deepcopy

from Illuminant.illuminant import Illuminant
from Utils.utils import read_spectral, rgb_to_xw, xw_to_rgb

class Macbeth():
    # def __init__(self,
    #              illuminant: Illuminant,
    #              patch_size: int = 16):
    #     self.sphotons: np.ndarray = None
    #     self.wavelength: np.ndarray = None
    #     self.patch_size: int = patch_size
        
    #     # Illuminant photons
    #     illuminant_wave = illuminant.wave
    #     iphotons = illuminant.photons
        
    #     # Read the surface data from the file repository
    #     surface = sio.loadmat('data/surfaces/MacbethChart.mat')
    #     wave = surface['wavelength']
    #     reflectance = surface['data']
        
    #     x = np.squeeze(wave)
    #     y = reflectance

    #     # Interpolate and extrapolate scene reflectance according to the light wavelength
    #     new_reflectance = np.zeros((len(illuminant_wave), y.shape[1]))
    #     for i in range(y.shape[-1]):
    #         new_reflectance[..., i] = interp1d(x, y[..., i], kind = 'linear', bounds_error = False, fill_value = 0)(illuminant_wave)
        
    #     self.sphotons = (new_reflectance.T @ np.diag(iphotons.flatten())).reshape(4, 6, -1, order = 'F')
    #     # self.energy = self.energy.reshape((-1, len(illuminant_wave)), order = 'F')
        
        
    #     # Increase size of macbeth chart using patch size
    #     self.increase_image_size()
                
    #     self.wavelength = illuminant_wave
    
    def __init__(self,
                 illuminant: Illuminant,
                 patch_size: int = 16):
        self.sphotons: np.ndarray = None
        self.wavelength: np.ndarray = None
        self.patch_size: int = patch_size
        
        # Illuminant photons
        self.wavelength = illuminant.wave
        iphotons = illuminant.photons
        n_wave = len(self.wavelength)
        
        # Read the surface data from the file directory
        macbeth_chart = read_spectral('data/surfaces/MacbethChart.mat', self.wavelength)  # shape: (31, 24)
        self.reflectance = macbeth_chart.T.reshape((4, 6, n_wave), order = 'F')
        
        
        # Increase size of macbeth chart using patch size
        self.increase_image_size()
        
        reflectance, r, c, _ = rgb_to_xw(self.reflectance)
        sphotons = reflectance @ np.diag(np.squeeze(iphotons))
        sphotons = xw_to_rgb(sphotons, r, c)
        
        self.sphotons = sphotons
        
    def increase_image_size(self) -> None:
        if self.reflectance.ndim == 3:
            w = self.reflectance.shape[-1]
        elif self.reflectance.ndim == 2:
            w = 1
        else:
            ValueError('Unexpected input matrix dimension')
            
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)
        
        r, c = self.reflectance.shape[:2]
        new_image = np.zeros((r * self.patch_size[0], c * self.patch_size[1], w))
        
        for i in range(w):
            new_image[:, :, i] = np.kron(self.reflectance[:, :, i], np.ones(self.patch_size))
            
        self.reflectance = deepcopy(new_image)
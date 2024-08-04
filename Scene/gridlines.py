import numpy as np

from Illuminant.illuminant import Illuminant

class Gridlines():
    """
    Scene comprising of an array of girdlines
    
    Args:
        - sz: size of the scene for one side
        - line_spacing: distance between adjacent lines
        - line_thickness: thickenss of the lines to be drawn
        - illuminant: light source
        
    Returns:
        - Gridlines sturcture itself
        
    Description:
        The grid line scene is useful for visualizaing the geometric distortion of a lens. 
        The spectral power distribution (SPD) of the lines is set to D65 unless specified otherwise
    """
    
    def __init__(self,
                 sz: int = 128, 
                 line_spacing: int = 16,
                 line_thickness: int = 1,
                 illuminant: Illuminant = None):
        self.wavelength = np.arange(400, 701, 10)
        n_wave = len(self.wavelength)
        
        if isinstance(sz, int):
            sz = (sz, sz)
        
        d = np.zeros(sz)
        
        for i in range(line_thickness):
            d[(np.arange(line_spacing // 2, sz[0], line_spacing) + i)[:, None], :] = 1
            d[:, (np.arange(line_spacing // 2, sz[1], line_spacing) + i)] = 1

        d[d == 0] = 1e-5
            
        reflectance = np.repeat(d[:, :, np.newaxis], n_wave, axis = 2)
        
        self.sphotons = reflectance * illuminant.photons
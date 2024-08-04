import numpy as np
from itertools import product

from Illuminant.illuminant import Illuminant

class PointArray(): 
    """
    Make a point array stimulus for evaluating the optics
    
    The point array scene clarifies the PSF at a variety of locations in the optical image.
    It also gives a sense of the geometric distortion in the optical image
    
    Args:
        - wave: wavelength of the light
        - sz: total size of the point array scene. it is the length of one side of the square scene
        - point_spacing: horizontal or vertical distance between adjacent points in the point array scene
    
    Returns:
        - point_array: PointArray struct
    """

    def __init__(self, 
                 wave : np.ndarray = np.arange(400, 701, 10), 
                 sz : int = 128, 
                 point_spacing : int = 16,
                 illuminant : Illuminant = None):
        self.wave = wave
        
        if not isinstance(sz, int):
            raise ValueError("sz must be an integer")

        space_array = np.zeros((sz, sz))
        point_location = np.arange(round(point_spacing / 2) - 1, sz, point_spacing)
        for i, j in product(point_location, point_location):
            space_array[i, j] = 1
        
        reflectance = np.repeat(space_array[:, :, np.newaxis], len(wave), axis = 2)
        
        self.sphotons = reflectance * illuminant.photons
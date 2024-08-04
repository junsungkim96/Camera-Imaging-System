from typing import Tuple
import numpy as np

from Illuminant.illuminant import Illuminant

class Slantededge():
    """
    
    
    """
    def __init__(self,
                 image_size : int = 384,
                 bar_slope : float = 2.6,
                 fov: int = 2,
                 wave: np.ndarray = np.arange(400, 701, 10),
                 dark_level: float = 0,
                 illuminant: Illuminant = None):
        self.wavelength = wave
        n_wave = len(self.wavelength)
        
        image = self.slanted_edge(image_size, bar_slope, dark_level)
        
        image = np.clip(image, 1e-6, 1)
        
        photons = image[..., np.newaxis] * illuminant.photons
            
        self.sphotons = photons
        self.wangular = fov
        
    def slanted_edge(self,
                     image_size: Tuple[int, int] = (384, 384), 
                     slope: float = 2.6,
                     dark_level: float = 0):
        """
        Make a slanted edge. Always square and odd number of rows/cols
        
        Args:
            - image_size: (row, col) (384, 384) by default
            - slope: slope of the edge
            - dark_level: dark side is set 0 by default. White side is always 1
            
        Returns:
            - slanted_edge structure itself
            
        Description:
            The target is used for ISO 12233 standard.
            By construction, the image size is always returned as odd
            The bright side is always 1. The dark level is a parameter
            
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        image_size = (np.round(image_size[0] / 2), np.round(image_size[1] / 2))
        
        x, y = np.meshgrid(np.arange(-image_size[1], image_size[1]+1), np.arange(-image_size[0], image_size[0]+1))
        
        image = np.ones(x.shape) * dark_level
        
        mask = (y > slope * x)
        
        image[mask] = 1
        
        return image
    
    

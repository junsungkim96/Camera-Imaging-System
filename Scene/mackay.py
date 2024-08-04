import numpy as np

from Illuminant.illuminant import Illuminant

class Mackay():
    """
    
    """
    def __init__(self,
                 radial_freq: int = 8,
                 sz: int = 256,
                 illuminant: Illuminant = None):
        
        self.wavelength = np.arange(400, 701, 10)
        n_wave = len(self.wavelength)
    
        image = self.image_mackay(radial_freq, sz)
    
        r = np.round(2 * radial_freq / np.pi)
        
        X, Y = np.meshgrid(np.arange(sz), np.arange(sz))
        X = X.astype('float64') - np.mean(X)
        Y = Y.astype('float64') - np.mean(Y)
        d = np.sqrt(X**2 + Y**2)
        
        image[d < r] = 128
        
        self.sphotons = np.repeat(image[:, :, np.newaxis], n_wave, axis = 2) * illuminant.photons
        
        return
            
    
    
    def image_mackay(self,
                     radial_freq: int = 8,
                     sz: int = 128):
        """
        Create a Mackay chart spatial pattern
        
        Args:
            - radial_freq: spatial frequency in the radial direction
            - sz: size of the chart
            
        Returns:
            - image: Mackay image
            
        Description:
            The Mackay chart has lines at many angles and increases in spatial frequency from periphery to center.
            
        """
        mx = np.round(sz / 2)
        mn = -(mx -1)
        
        x, y = np.meshgrid(np.arange(mn, mx+1), np.arange(mn, mx+1))
        x[x == 0] = np.finfo(float).eps  # Handle division by zero
        image = np.cos(np.arctan(y / x) * 2 * radial_freq)
        image, _, _ = self.scale_data(image, 1, 256)
        
        return image
        
    def scale_data(self, im, b1=None, b2=None):
        """
        Scale the values in `im` into the specified range [b1, b2].

        Args:
            im (np.ndarray): Input data.
            b1 (float, optional): Lower bound of the target range.
            b2 (float, optional): Upper bound of the target range.

        Returns:
            np.ndarray: Scaled data.
            float: Minimum value of the original data.
            float: Maximum value of the original data.
        """
        mn = np.min(im)
        mx = np.max(im)

        if b1 is None and b2 is None:
            # No bounds arguments, scale to [0, 1]
            b1, b2 = 0, 1
        elif b2 is None:
            # If only one bound is provided, it is interpreted as the maximum value
            b2 = b1
            b1 = 0

        # Scale the data to [0, 1]
        scaled_im = (im - mn) / (mx - mn)

        # Adjust to the specified range [b1, b2]
        range_diff = b2 - b1
        scaled_im = range_diff * scaled_im + b1

        return scaled_im, mn, mx
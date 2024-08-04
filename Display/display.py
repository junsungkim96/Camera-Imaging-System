import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Iterable
from collections import defaultdict
import reprlib
from dataclasses import dataclass
from color import color_block_matrix
import cv2
from scipy.ndimage import zoom



@dataclass
class Dixel:
    control_map: np.ndarray
    intensity_map: np.ndarray
    n_pixels: list[int, int]

class Display():
    """
    Create a display object.


    Display(d) calibration data are stored in a display instance.
    these are the spectral radiance distribution of its primaries and a gamma function.

    Args:
        display_name(str): Name of a file containing a calibrated display structure.
        the supported display files are stored in data/display.
        the files should contain a variable('d') as display structure.

        *kwargs: (Optional) User defined parameter values, should be in dictionary format.
        See the optional key/value pairs section.

    Attributes:
        name: Name of the display file


    Properties:
        whitespd(np.ndarray): White point spectral power distribution
        
    Examples:
        >>> Display()
        Display(name(LCD-Apple.mat), wave([380, 3... dtype=uint16), dpi(96), dist(0.5)...)
        >>> Display('LCD-Apple.mat',)
        Display(name(LCD-Apple.mat), wave([380, 3... dtype=uint16), dpi(96), dist(0.5)...)
    """

    def __init__(self,
                 display_name: str = 'LCD-Apple.mat',
                 wave: np.ndarray | None = None,
                 gamma: np.ndarray | None = None,
                 spd: np.ndarray | None = None,
                 dpi: int | None = None,
                 dist: float | None = None,
                 is_emissive: bool | None = None,
                 **kwargs):

        self.name = display_name

        self.image = None  # we will set it later

        match self.name:
            case 'default':
                # TODO: We will implement this later
                """
                Create a default display that works well with the imageSPD rendering routine.
                    See vcReadImage for more notes.
                """
                self.wave = np.arange(400, 701, 10)
                # Matlab `spd = pinv(colorBlockMatrix(length(wave))) / 700;`
                self.spd = np.linalg.pinv(color_block_matrix(self.wave)) / 700
                N = 256
                g = np.tile(np.linspace(0, 1, N), (3, 1)).T
                self.gamma = g
                self.dpi = 96  # typical display density
                self.dist = 0.5  # typical viewing distance
                raise NotImplementedError('Default display is not implemented yet')

            case 'equalenergy' | 'equal energy':
                raise NotImplementedError('Equal energy display is not implemented yet')
                # self.default_display()
                # spd = np.ones(self.spd.shape) * 1e-3
                # self.spd = spd

            case _:
                base_dir = Path(__file__).parent.parent / 'data/displays'
                if (base_dir / self.name).exists():
                    tmp = loadmat(str(base_dir / display_name))['d'][0, 0]
                    self.wave = wave if wave is not None else tmp['wave'].flatten()
                    self.gamma = gamma if gamma is not None else tmp['gamma']
                    self.spd = spd if spd is not None else tmp['spd']
                    self.dpi = dpi or tmp['dpi'][0][0]
                    self.dist = dist or tmp['dist'][0][0]
                    self.is_emissive = bool(is_emissive or tmp['isEmissive'][0][0])
                    data_dixel = tmp['dixel'][0][0]
                    self.dixel = Dixel(data_dixel['controlmap'], data_dixel['intensitymap'], data_dixel['nPixels'])

                else:
                    raise ValueError('Unknown display type')

        self.whitespd = np.sum(self.spd, axis=1, keepdims=True)

    @property
    def n_primaries(self):
        return self.spd.shape[-1]

    @property
    def meter_per_dot(self):
        dpi = self.dpi
        dist = self.dist
        val = np.atan()

    @property
    def over_sample(self):
        


    def __repr__(self) -> str:
        components = reprlib.repr(self.wave)
        components = components[components.find('['):-1]
        return f'Display(name({self.name}), wave({components}), dpi({self.dpi}), dist({self.dist})...)'

    def compute(self, image: np.ndarray, sz: tuple[int, int]) -> np.ndarray:
        """
        Compute the display image from the input image.

        Args:
            image: Input image
            sz: Size of the display image

        Returns:
            display_image: Display image
        """
        
        if image is None:
            raise ValueError('Input image required')
        
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED).astype(np.float64) / 255.0
        else:
            image = image.astype(np.float64)
            
        n_primary = self.n_primaries
        
        # If no upsampling, then s is the size of the psf
        if sz is None:
            s = self.over_sample
            dixel_img = self.dixel_image()
            sz = self.dixel_size
        else:
            s = np.round(np.array(sz) / self.pixels_per_dixel).astype(int)
            dixel_img = self.dixel_image(sz)
            if not all(s > 0):
                raise ValueError('Bad up-sampling size')
            
        # Check psf values to be no less than 0
        if dixel_img is None:
            raise ValueError('psf not defined for display')
        if np.min(dixel_img) < 0:
            raise ValueError('psfs values should be non-negative')
        
        # If a single matrix, assume it is gray scale
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], n_primary, axis = 2)
            
        # Expand the image so there are s samples within each of the pixels allowing a representation of a psf
        M, N, _ = image.shape
        ppd = self.pixel_per_dixel
        h_render = self.render_function
        
        if np.any(ppd > 1) and h_render is None:
            raise ValueError('Render algorithm is required')
        
        if h_render is not None:
            out_image = h_render(image, sz)
        else:
            out_image = zoom(image, (s[0], s[1], 1), order = 0)
            
        # Check the size of out_image
        if out_image.shape[0] != M * s[0] or out_image.shape[1] != N * s[1]:
            raise ValueError('Bad out image size')
        
        out_image *= np.tile(dixel_img, (M // ppd[0], N // ppd[1], 1))
        
        return out_image
        
        

if __name__ == '__main__':
    display_cal_file = 'LCD-Apple.mat'
    display_cal_file = 'OLED-Sony.mat'
    display = Display(display_cal_file)
    print(display)

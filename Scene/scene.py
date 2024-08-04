# Python built-in modules
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pip installed packages
import numpy as np
from skimage.color import xyz2rgb
import skimage
import matplotlib.pyplot as plt

# Local modules
from Illuminant.illuminant import Illuminant
from Utils.utils import energy_to_quanta, quanta_to_energy, unit_conversion, read_spectral, rgb_to_xw, xw_to_rgb, xyz_from_energy
from Scene.point_array import PointArray
from Scene.macbeth import Macbeth
from Scene.slantededge import Slantededge
from Scene.gridlines import Gridlines
from Scene.mackay import Mackay

class Scene():
    """
    A scene describes the photons emitted from each visible point in the scene.
    Generally, we model planar objects, such as a screen display but plan to add 3D scenes with various depths
    
    The scene is located at some distance from the center of the optics, has a field of view, and a spectral radiance distribution.
    There are routines to handle depth as well as that partly implemented and under development. 
    Plan to integrate this aspect with PBRT(Physics Based Ray Tracing) 
    
    A variety of scene types can be created automatically. The routines that create these scenes serve as a template for creating
    others that you wish to design.
    
    Scene radiance data are represented as photons single precision. The spectral representation is np.arange(400, 701, 10) by default. 
    Both of these can be changed
    
    Args:
        - scene_name: Name of the scene type to create
        - illuminant: Illuminant struct created as the light source
        - sz(point_array): total size of the point array scene. it is the length of one side of the square scene
        - point_spacing(point_array): horizontal or vertical distance between adjacent points in the point array scene
        - surface_file: directory of the scene file that is read in 
        - patch_size(macbeth_chart): one side length of a single patch for 24 color patch Macbeth color chart
        - patch_list(macbeth_chart): number of pathces that would be used from 24 color of Macbeth chart
        - black_border(macbeth_chart): whether to include or not include a black border between the patches in the Macbeth color chart
        - scene_distance: the distance from the center of the scene to the center of the optics
        - hfov: horizontal field of view of the scene to the optics 
        
    Returns
        - scene: Scene structure
    
    """
    def __init__(self,
                scene_name: str,
                illuminant: Illuminant | None = None,
                # Point array parameters
                sz : int = 128, point_spacing : int = 16, *,
                # Macbeth chart parameters
                patch_size = 16, black_border = False,
                # Common parameters
                surface_file = None, scene_distance = 1.2, hfov = 10):

        # Parameters that are set by the user. All the other parameters are derived from these two parameters
        self.distance = scene_distance
        self.wangular = hfov
        
        # The followings are the parameters used later on. Magnification is always set to 1
        self.magnification: int = 1.0  # Not used
        self.scene_name: str = scene_name
        self.surface_file: str = surface_file
        self.wavelength: np.ndarray = None
        self.energy: np.ndarray = None
        self.sphotons: np.ndarray = None
        self.xyz = None
        self.rgb = None
        
        scene_name = scene_name.lower().replace(" ", "")
        
        match scene_name:
            ## Point Array
            case 'pointarray':
                point_array = PointArray(sz = sz, point_spacing = point_spacing, illuminant = illuminant)
                self.wavelength = point_array.wave
                self.sphotons = point_array.sphotons
                # self.wangular = point_array.wangular
            ## Macbeth Chart
            case 'macbeth' | 'macbethchart':    
                macbeth = Macbeth(patch_size = patch_size, illuminant = illuminant)
                self.wavelength = macbeth.wavelength
                self.sphotons = macbeth.sphotons
            ## Gridlines
            case 'gridlines':
                gridlines = Gridlines(illuminant = illuminant)
                self.wavelength = gridlines.wavelength
                self.sphotons = gridlines.sphotons
            ## SlantedEdge
            case 'slantededge' | 'iso12233':
                slantededge = Slantededge(illuminant = illuminant, image_size = 384)
                self.wavelength = slantededge.wavelength
                self.sphotons = slantededge.sphotons
            ## Rings and rays
            case 'ringsrays':
                mackay = Mackay(illuminant = illuminant)
                self.wavelength = mackay.wavelength
                self.sphotons = mackay.sphotons
        
        # Adjust the luminance to the target luminance
        self.adjust_luminance(target_luminance = 100)

    def adjust_luminance(self, target_luminance):
        # Convert photons into energy
        self.energy = quanta_to_energy(self.wavelength, self.sphotons)
        
		# Calculate the current luminance from energy and wave
        current_luminance = self.calculate_luminance()
  
		# Adjust the energy based on target luminance and current luminance
        new_energy = self.energy * (target_luminance / current_luminance.mean())
  
        # Update photons and energy to the new luminance
        self.sphotons = energy_to_quanta(self.wavelength, new_energy)
        self.energy = new_energy
        
        # Convert energy into XYZ values
        self.xyz = xyz_from_energy(self.energy, self.wavelength)

        # Convert XYZ tristimulus values into RGB values
        self.rgb = xyz2rgb(self.xyz)
        self.rgb = skimage.util.img_as_ubyte(self.rgb)
        
        return
    
    def calculate_luminance(self):
        V = read_spectral('data/human/luminosity', self.wavelength)
        
        xw_data, r, c, _ = rgb_to_xw(self.energy)
        
        luminance = 683 * (xw_data @ V) * (self.wavelength[1] - self.wavelength[0])
        luminance = luminance[:, np.newaxis] 
    
        luminance = xw_to_rgb(luminance, r, c)
        
        return luminance
    
    def display(self) -> None:        
        plt.imshow(self.rgb)
        plt.xticks([])
        plt.yticks([])
        plt.show()


    @property
    def rows(self):
        return self.sphotons.shape[0]

    @property
    def cols(self):
        return self.sphotons.shape[1]
    
    @property
    def size(self):
        return [self.rows, self.cols]
    
    @property
    def hangular(self):
        h = self.height()
        d = self.distance
        val = np.degrees(2 * np.arctan((0.5 * h) / d))
        
        return val

    @property
    def diagonal_fov(self):
        """"

        """
        vd = self.distance
        rw = np.deg2rad(self.wangular)
        rh = np.deg2rad(self.hangular)
        d = np.sqrt((vd * np.tan(rw))**2 + (vd * np.tan(rh))**2)
        
        val = np.rad2deg(np.arctan2(d, vd))
        
        return val

    def sample_spacing(self, units = 'm'):
        sz = self.size
        val = [self.width() / sz[1], self.height() / sz[0]]

        if units:
            val *= unit_conversion(units)
            
        return val

    def sample_size(self, units = 'm'):
        w = self.width()
        c = self.cols
        val = w / c
        
        if units:
            val *= unit_conversion(units)
        
        return val

    def height(self, units = 'm'):
        s = self.sample_size()
        r = self.rows
        val = s * r
        
        if units:
            val *= unit_conversion(units)
        
        return val

    def width(self, units = 'm'):
        d = self.distance
        w = self.wangular
        val = 2 * d * np.tan(np.deg2rad(w / 2))
        
        if units:
            val *= unit_conversion(units)
        
        return val


if __name__ == '__main__':
    import scipy.io as sio
    
    ## macbeth ##
    # illuminant = Illuminant(illuminant_name = 'd65')
    # scene = Scene(scene_name = 'macbeth', illuminant = illuminant, scene_distance = 1.2, hfov = 10)
    # tmp = sio.loadmat('data_validation/macbeth_photons.mat')['photons']
    # print(np.allclose(scene.sphotons, tmp, rtol = 1e-04))
    # scene.display()
    
    ## point_array ##
    # illuminant = Illuminant(illuminant_name = 'tungsten')
    # scene = Scene(scene_name = 'pointarray', illuminant = illuminant)
    # tmp = sio.loadmat('data_validation/point_array_photons.mat')['photons']
    # print(np.allclose(scene.sphotons, tmp, rtol = 1e-04))
    # scene.display()
    
    ## Gridlines
    # illuminant = Illuminant(illuminant_name = 'd65')
    # scene = Scene(scene_name = 'gridlines', illuminant = illuminant)
    # scene.display()
    
    ## Slantededge
    # illuminant = Illuminant(illuminant_name = 'd65')
    # scene = Scene(scene_name = 'slantededge', illuminant = illuminant)
    # scene.display()
    
    ## Mackay
    illuminant = Illuminant(illuminant_name = 'tungsten')
    scene = Scene(scene_name = 'ringsrays', illuminant = illuminant)
    scene.display()
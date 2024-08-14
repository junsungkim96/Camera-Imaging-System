import numpy as np
import scipy.io as sio
import imageio
from skimage import img_as_float
from scipy.interpolate import interp1d

import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

from Scene.sceneRGB import sceneRGB
from Scene.sceneMultispectral import sceneMultispectral
from Illuminant.blackbody import blackbody
from Illuminant.illuminant import Illuminant
from Scene.scene import Scene
from Display.display import Display
from Utils.utils import energy_to_quanta, rgb_to_xw, xw_to_rgb


class sceneFromFile(Scene):
    """
    Create a scene structure by reading data from a file
    
    Args:
        - input_data: Typically, this is the name of an RGB image file. But, it may also be RGB data.
        - image_type: 'spectral', 'rgb' or 'monochrome'
        - mean_luminance: If a value is sent in, set scene to this mean_luminance.
        - disp_cal: A display structure used to convert RGB to spectral data.
                    For the typical case such as an emissive display, the illuminant SPD is modeled and set to the white point of the display
        - wave_list: The scene wavelength samples.
        - illuminant_energy: Use this as the illuminant energy. It must have the same wavelength sampling as wave_list
        - scale_reflectance: Adjust the illuminant_energy level so that the maximum reflectance is 0.95. Default: True.
    
    Returns:
        - scene: scene structrue
        
    Description:
        The data in the image file is converted into spectral format and placed as scene data structure. The allowable image types are
        monochrome, rgb, multispectral and hyperspectral. If not specified and cannot be inferred, it might be asked. 
        
        If the image is in RGB format, a display calibration file(disp_cal) may be specified. This file contains display calibration data
        that are used to convert the RGB values into a spectral radiance image. If disp_cal is not defined, the default display file 'lcdExample'
        will be used. 
        
        The default illuminant for a RGB file is the display white point.  
    
    """
    def __init__(self,
                 input_data: str | None = None, 
                  image_type: str | None = None, 
                  mean_luminance: int | None = None, 
                  disp_cal = None,
                  wave_list = None, 
                  illuminant_energy = None, 
                  scale_reflectance = True):
        super().__init__()
    
    
        ## Parameter Setup
        # Initialize scene as None
        scene = None

        # Check if illuminant_energy is defined
        if illuminant_energy is None:
            illuminant_energy = []

        if input_data is None:
            raise ValueError("Input data is required")

        if isinstance(input_data, str):
            # Read the file based on its extension
            file_ext = input_data.split('.')[-1].lower()
            if file_ext in ['mat']:
                mat_contents = sio.loadmat(input_data)
                if 'scene' in mat_contents:
                    return mat_contents['scene']
            elif file_ext in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                input_data = imageio.v2.imread(input_data)

        ## Determine the photons and illuminant structure
        
        # Determine the image type if not defined
        if image_type is None:
            raise ValueError("Image type is required")
        image_type = image_type.lower()

        # Handle different image types
        match image_type:
            case 'monochrome' | 'rgb':
                # Initialize display structure if not provided
                if disp_cal is None:
                    disp_cal = Display()

                # Load the display calibration file if it's a path
                if isinstance(disp_cal, str):
                    theDisplay = Display(disp_cal)
                elif isinstance(disp_cal, Display) and disp_cal.type == 'display':
                    theDisplay = disp_cal
                else:
                    raise ValueError("Bad display information.")

                # Set the display wavelength if wave_list is provided
                if wave_list is not None:
                    spd = theDisplay.spd
                    wave = theDisplay.wave
                    
                    new_spd = np.zeros((len(wave_list), spd.shape[1]))
                    for i in range(spd.shape[1]):
                        new_spd[:, i] = interp1d(wave, spd[:, i], kind = 'linear')(wave_list)
                    
                    
                    
                    am = np.zeros(theDisplay.wave.shape)
                    new_am = interp1d(wave, am, kind = 'linear', fill_value = 0)(wave_list)
                    
                    theDisplay.wave = wave_list
                    theDisplay.spd = new_spd
                    theDisplay.am = new_am
                    
                wave = theDisplay.wave
                
                do_sub = False
                sz = None

                # Get the scene spectral radiance using the display model
                photons = self.read_image_file(input_data, image_type, theDisplay, do_sub, sz)

                ## Initialize scene
                # Match the display wavelength and the scene wavelength
                scene = sceneRGB()
                scene.wave = wave

                # This code handles both emissive and reflective displays. The white point is set a little differently.
                # (a) For emissive display, set the illuminant SPD to the white point of the display if ambient
                #     lighiting is not set.
                # (b) For reflective display, the illuminant is required and should be passed in as *args

                # Initialize the whole illuminant struct
                if len(illuminant_energy) == 0:
                    if not theDisplay.is_emissive:
                        raise ValueError("Illuminant energy specification required for reflective display")
                    else:
                        illuminant_energy = np.sum(theDisplay.spd, axis=1)

                il = Illuminant('D65', wave)
                illuminant_energy
                
                wave = il.wave
                if illuminant_energy.ndim > 2:
                    illuminant_energy, r, c, _ = rgb_to_xw(illuminant_energy)
                    il_photons = energy_to_quanta(wave, illuminant_energy.T).T
                    il_photons = xw_to_rgb(il_photons, r, c)
                    il.photons = il_photons
                else:
                    il.photons = energy_to_quanta(wave, illuminant_energy)
                
                # Compute photons for reflective display
                # For reflective display, until this step, the photon variable stores reflectance information
                if not theDisplay.is_emissive:
                    il_photons = il.photons
                    il_photons = il_photons.reshape(1, 1, len(wave))
                    photons = photons * il_photons

                # Set viewing distance and field of view
                if hasattr(theDisplay, 'distance'):
                    scene.distance = theDisplay.distance
                else:
                    scene.distance = 0.5

                img_size = photons.shape[1]
                img_fov = img_size * theDisplay.deg_per_dot
                scene.wangular = img_fov
                scene.distance = theDisplay.viewing_distance
                
            case 'spectral' | 'multispectral' | 'hyperspectral':
                if not isinstance(input_data, str):
                    raise ValueError("Name of existing file required for multispectral")
                if wave_list is None:
                    wave_list = []

                scene = sceneMultispectral()

                # Read image and illuminant structure
                photons, il, basis, _, _ = self.read_image_file(input_data, image_type, wave_list)

                # Override the default spectrum with the basis function wavelength sampling
                scene.wavelength = basis.wave

                if isinstance(input_data, str) and 'name' in sio.whosmat(input_data):
                    mat_contents = sio.loadmat(input_data)
                    scene.name = mat_contents['name']

            case _:
                raise ValueError("Unknown image type")

        ## Put the remaining parameters in place and return
        
        if isinstance(input_data, str):
            scene.filename = input_data
        else:
            scene.filename = 'numerical'

        scene.photons = photons
        scene.illuminant = il

        # Name the scene with the filename or just announce that we received rgb data.
        # Also check whether the file contains 'fov' and 'dist' variables, adjust the scene, override the data.
        if isinstance(input_data, str):
            n = input_data.split('/')[-1].split('.')[0]
            if input_data.endswith('.mat'):
                mat_contents = sio.loadmat(input_data)
                if 'fov' in mat_contents:
                    scene.fov = mat_contents['fov']
                if 'dist' in mat_contents:
                    # scene = sceneSet(scene, 'distance', mat_contents['dist'])
                    scene.distance = mat_contents['dist']
        else:
            n = 'rgb image'

        if 'theDisplay' in locals():
            n = f"{n} - {theDisplay.name}"
        scene.name = n

        if mean_luminance is not None:
            scene = scene.adjust_luminance(mean_luminance)

        # Adjust illuminant level to a max reflectance 0.95.
        # If the reflectances are expected to be dark, set to 'False'
        if scale_reflectance:
            r = scene.reflectance
            max_reflectance = np.max(r)
            if max_reflectance > 0.95:
                illuminant_energy = scene.illuminant_energy
                scene.illuminantenergy = illuminant_energy * max_reflectance



    def read_image_file(self, fullname, image_type = 'rgb', *args):
        """
        Read in monochrome, rgb, or multispectral data and return multispectral photons
        
        The image data in fullname are converted into photons. The other parameters can be returned if needed. 
        
        There are several different image file types. Image file types are usually figured out from othe file name. 
        If that fails, the user is queried. 
        
        Args:
            - fullname: Either a file name or possible RGB data read from a file.
            - image_type: There are two types. 
                        - 'rgb', 'unispectral', 'monochrome': The data in RGB or other format are returned as photons
                                                                estimated by putting the data into the display framebuffer 
                        - 'spectral', 'multispectral', 'hyperspectral': The data are stored as coefficients and and basis functions.
                                                                        Spectral representation, comment, measurement of the scene 
                                                                        illuminant can be returned. 

        Returns:
            - photons: RGB format of photon data (r, c, w)
            - illuminant: An illuminant class
            - basis: Structure containing basis functions for multispectral SPD
            - comment
            - mc_coef: Coefficients for basis functions for multispectral SPD
        """
        # These are loaded for a file, when they are returned
        mc_coef = None
        comment = ''
        
        
        image_type = image_type.lower().replace(" ", "")
        
        match image_type:
            case 'rgb' | 'unispectral' | 'monochrome':
                disp_cal = args[0] if len(args) > 0 else None
                do_sub = args[1] if len(args) > 1 else False
                sz = args[2] if len(args) > 2 else None
                
                # Scene rendered without subpixel (do_sub = False) could be different from the one rendered with subpixel of size [m n]
                # where m and n indicate the number of pixels per dixel in rows and columns. This kind of inconsistency will occur 
                # especially when pixels per dixel is not [1, 1].
                
                # To be more accurate, please turn off subpixel rendering if the samples per dixel gets very small. 
                
                if isinstance(fullname, str):
                    in_img = np.double(plt.imread(fullname))
                else:
                    in_img = np.double(fullname)
                
                # An rgb image    
                if disp_cal is None:
                    if in_img.ndim == 2:
                        in_img = np.repeat(in_img[:, :, np.newaxis], 3, axis = 2)
                    if in_img.ndim != 3:
                        raise ValueError(f'Bad number of dimensions {in_img.ndim} of image')
                    
                    # If there is no display calibration file, we arrange the photon values so that the scene window shows 
                    # the same RGB values as in the original file. 
                    
                    print('Using block matrix primaries')
                    xw_img = rgb_to_xw(in_img / 255)
                    xw_img = np.clip(xw_img, 1e-3, 1)
                    
                    # When we render the RGB data in xw_img, they are multiplied by the color_block_matrix. 
                    # By storing the photons this way, the displayed image in the scene will be the same as the original RGB image. 
                    photons = xw_img @ np.linalg.pinv(self.color_block_matrix(31))
                else:
                    if isinstance(disp_cal, str):
                        the_display = Display(disp_cal)
                    elif isinstance(disp_cal, Display) and disp_cal.type == 'display':
                        the_display = disp_cal
                    else:
                        raise ValueError('Unknown display structure')
                    
                    wave = the_display.wave
                    g_table = the_display.gamma
                    n_primaries = the_display.n_primaries
                    
                    if in_img.ndim == 2:
                        in_img = np.repeat(in_img[:, :, np.newaxis], n_primaries, axis = 2)
                        
                   
                    in_img = np.pad(in_img, ((0, 0), (0, 0), (0, n_primaries - in_img.shape[2])), 'constant', constant_values = 0)
                        
                    if in_img.shape[2] != n_primaries:
                        raise ValueError('Bad image size')
                        
                    if in_img.max() > g_table.shape[0]:
                        img_max = in_img.max()
                        n_bits = int(np.floor(np.log2(img_max)))
                        warnings.warn(f'Input data ({n_bits}) exceeds gTable ({int(np.log2(g_table.shape[0]))}bits)')
                        if n_bits >= 14:
                            max_display = 2 ** 16
                        elif n_bits >= 12:
                            max_display = 2 ** 14
                        elif n_bits >= 10:
                            max_display = 2 ** 12
                        elif n_bits >= 8:
                            max_display = 2 ** 10
                        
                        x_in = np.linspace(0, 1, g_table.shape[0])
                        y_in = g_table
                        x_out = np.linspace(0, 1, (2 ** (n_bits + 1))) / (2 ** (n_bits + 1) - 1)
                        g_table = interp1d(x_in, y_in, kind = 'spline')(x_out)
                        in_img = in_img / img_max    
                        in_img = np.round(in_img  * (g_table.shape[0] - 1))
                    
                    elif in_img.max() <= 1:
                        in_img = np.round(in_img * (g_table.shape[0] - 1))
                    
                    elif in_img.max() <= 255:
                        s = g_table.shape[0]
                        
                        if s > 256: 
                            in_img = np.round(in_img / 255 * (s - 1))
                            
                    in_img = self.lut_digital(in_img, g_table)
                    
                    if do_sub:
                        in_img = the_display.compute(in_img, sz)
                        
                    spd = the_display.spd
                    xw_img, r, c, _ = rgb_to_xw(in_img)
                    
                    if xw_img.size < 1e7:  # Use threshold as needed
                        photons = energy_to_quanta(wave, (xw_img @ spd.T).T).T
                    else:
                        photons = np.zeros((xw_img.shape[0], max(wave.shape)))
                        for i in range(max(wave.shape)):
                            photons[:, i] = energy_to_quanta(wave[i], (xw_img @ spd[i, :].T).T).T
                            
                photons = xw_to_rgb(photons, r, c)
                
                return photons
                
            case 'spectral', 'multispectral', 'hyperspectral':
                variables = loadmat(fullname)
                if 'mcCOEF' in variables:
                    mc_coef = variables['mcCOEF']
                    basis = variables['basis']
                    comment = variables.get('comment', '')
                    
                    if len(args) > 0 and args[0] is not None:
                        old_wave = basis['wave']
                        new_wave = args[0]
                        extrap_val = 0
                        n_bases = basis['basis'].shape[1]
                        new_bases = np.zeros((max(new_wave.shape), n_bases))
                        for i in range(n_bases):
                            new_bases[:, i] = np.interp(old_wave.reshape(-1, 1), basis['basis'][:, i].reshape(-1, 1), \
                                                        left = extrap_val, right = extrap_val)(new_wave)
                        basis['basis'] = new_bases
                        basis['wave'] = new_wave

                    # The image data should be in units of photons
                    photons = self.image_linear_transform(mc_coef, basis['basis'].T)
                    
                    if 'imgMean' in variables:
                        img_mean = variables['imgMean']
                        if len(args) > 0 and args[0] is not None:
                            extrap_val = 0
                            img_mean = np.interp(old_wave.reshape(-1, 1), img_mean.reshape(-1, 1),
                                                left = extrap_val, right = extrap_val)(new_wave)
                        
                    photons, r, c, _ = rgb_to_xw(photons)
                    
                    try:
                        photons = np.tile(img_mean.reshape(-1, 1), (1, r*c)) + photons.T
                    except MemoryError:
                        print('Memory Error')
                        
                    photons = xw_to_rgb(photons.T, r, c)
                        
                    illuminant = variables.get('illuminant', None)
                    if illuminant is None:
                        warnings.warn(f'No illuminant information in {fullname}')
                    
                    photons = np.max(photons, 0)
                
                else:
                    print('Reading multispectral data with raw data')
                    
                    photons = variables.get('photons', variables.get('data', None))
                    if photons is None:
                        raise ValueError('No photon data in file')
                    
                    comment = variables.get('comment', '')
                    wave = variables.get('wave', variables.get('wavelength', None))
                    if wave is None:
                        raise ValueError('No wavelength data in file')
                    
                    if len(args) > 0 and args[0] is not None:
                        new_wave = args[0]
                        perfect = 0
                        idx = self.find_wave_index(wave, new_wave, perfect)
                        photons = photons[:, :, idx]
                        wave = new_wave
                    
                    basis = {'basis': [], 'wave': np.round(wave)}
                
                illuminant = variables.get('illuminant', None)
                if illuminant is None:
                    warnings.warn(f'No illuminant information in {fullname}')
                
                # illuminant = illuminantModernize(illuminant)
                
                if len(args) > 0 and args[0] is not None:
                    illuminant.wave = new_wave
            
                return photons, illuminant, basis, comment, mc_coef
                    
            case _:
                raise ValueError(f'Unknown image type: {image_type}')
                
                return


    def color_block_matrix(self, w_list, extrap_value = 0):
        """
        Create a matrix to render SPD data into RGB
        
        We render spectral data in the scene and optical image windows as RGB images. 
        The matrix returned by this routine is used to calculate R, G and B values from the SPD. 
        The columns of the returned matrix define how to sum across the wavebands. 
        
        By default, the wavelengths from 400-490 add to the blue channel, from 500-570 add to the green channel,
        and 580-700 add to the red channel. Wavelengths outside of this band do not contribute. 
        
        It is possible to generate a contribution from outside the band, say in the infrared. When we try to 
        visualize IR, it is useful to set a value of 0.1 or 0.2. 
        
        Args:
            - w_list: the list of wavelengths in the SPD to be rendered
            - extrap_value: the amount contributed outside the visible band. Default : 0.
        
        Returns:
            - b_matrix: the color matrix used for rendering a photon spectrum 
                        e.g. dispaly_RGB = photon_SPD * b_matrix, photon_SPD: column vector
        """
        # if w_list is a single number, then we interpret it as one of the common visible wavelength ranges. 
        w_list = np.array([w_list])
        if max(w_list.shape) == 1:
            if w_list == 31: 
                w_list = np.arange(400, 701, 10)
            elif w_list == 371:
                w_list = np.arange(370, 731)
            elif w_list == 37:
                w_list = np.arange(370, 731, 10)
                
        # The default block matrix function is defined over 400:10:700 range
        default_W = np.arange(400, 701, 10)
        
        b = 10
        g = 8
        r = 31 - b - g
        
        default_matrix = np.vstack([
            np.concatenate([np.zeros(b), np.zeros(g), np.ones(r)]), 
            np.concatenate([np.zeros(b), np.ones(g), np.zeros(r)]),
            np.concatenate([np.ones(b), np.zeros(g), np.zeros(r)])
        ])
        
        default_matrix = default_matrix.T
        
        # Adjust for any differences in the wave list
        if np.array_equal(w_list, np.arange(400, 701, 10)):
            d = np.sum(default_matrix, axis = 0)
            b_matrix = default_matrix @ np.diag(1 / d)
        else:
            # Adjust the matrix to match the default over 400-700 but be a small value in the infrared. 
            # The default is 0 but could be non-zero value the user sends in 
            b_matrix = np.zeros(max(w_list.shape), 3)
            
            for i in range(3):
                b_matrix[:, i] = interp1d(default_W, default_matrix[:, i], kind = 'linear', fill_value = extrap_value, bounds_error = False)(w_list)
            
            d = np.sum(b_matrix)
            b_matrix = b_matrix @ np.diag(1 / d)
            
        wp = 'd65'
        match wp.lower():
            case 'ee':
                ee = np.ones(max(w_list.shape), 1)
                ee_photons = energy_to_quanta(w_list, ee)
                white_spd = ee_photons / np.max(ee_photons)
            case 'd65':
                d65_photons, _ = blackbody(w_list, 6500, 'photons')
                white_spd = d65_photons / np.max(d65_photons)
            case _:
                white_spd = None
                
        b_matrix = np.diag(1 / white_spd.flatten()) @ b_matrix
        
        return b_matrix
      
        
    def lut_digital(self, DAC, g_table = None):
        """
        Convert DAC values to linear RGB intensities through a gamma table
        
        The DAC values are digital values with a bit detph that is determined by the device. 
        We assume that the smallest DAC value is 0 and the largest value is 2 ^ (n_bits) -1.
        
        A g_table normally has size 2 ^ (n_bits) * 3, a table for each channel
        If size is 2 ^ (n_bits) * 1, we assume three channels are the same. 
        
        If the g_table is a single number, we raise the data to the power g_table.
        
        The g_table is normally stored in the display calibration files. 
        
        For this routine, the returned RGB values are in the range of [0, 1].
        They are linear with respect to radiance(intensity) of the display primaries. 
        """
        
        if g_table is None:
            g_table = 2.2
            
        if g_table.size == 1:
            RGB = DAC ** g_table
            return RGB
        
        if np.max(DAC) > g_table.shape[0]:
            raise ValueError(f'Max DAC value {np.max(DAC)} exceeds the row dimension {g_table.shape[0]} of the g_table')
        
        if np.max(g_table) > 1 or np.min(g_table) < 0:
            raise ValueError(f'g_table entries should be between 0 and 1.')
        
        # Convert through the table
        RGB = np.zeros(DAC.shape)
        g_table = np.tile(g_table, (1, DAC.shape[2]))
        
        for i in range(DAC.shape[2]):
            this_table = g_table[:, i]
            RGB[:, :, i] = this_table[DAC[:, :, i].astype(int)]
            
        return RGB


    def image_linear_transform(self, image, transform):
        """
        Apply a linear transformation to the channels of an RGB or XW image
        
        The image data (image) can be in the N x M x W format, 
        (e.g. W = 3 if RGB or W = 31 if the wavelength samples are np.arange(400, 701, 10)).
        The routine applies a right side multiplication to the data. Specifically, if an image
        point is represented by the row vector, p = [R G B] the matrix transforms each color point, p,
        to an output vector pT. In this case, T has 3 rows.
        
        The image can also be in XW format, in which case the transform is applied as XW * T,
        where T is expressed as a column vector. 
        
        If the data are viewed as wavelength samples, say [w1, w2, ... wn], 
        then the transform T must have n rows. 
        
        This routine works with colorTransformMatrix function, which provides access to 
        various standard color transformation matrices. 
        
        This routine works with image in the format (N x M x W) and a T matrix size (W x K),
        where K is the number of output channels. It also works with an image in XW format, 
        again with T in (W x K) format. 
        
        Args:
            - image: The original image in N x M x W format or XW format.
            - T: The transform to act upon the image
        
        Returns:
            - image_T : The transformed image
        """
        # Determine image format to convert to XW
        if image.ndim == 3 or (image.ndim == 2 and transform.shape[0] == 1):
            format = 'RGB'
            r, c, w = image.shape
            
            image, _, _, _ = rgb_to_xw(image)
        else:
            format = 'XW'
            _, w = image.shape
        
        if len(transform) != w:
            raise ValueError('image/transform data sizes mismatch')
        
        # Multiply and reformat back to RGB if necessary
        image_T = image @ transform
        
        if format == 'RGB':
            image_T = xw_to_rgb(image_T, r, c)
        
        return image_T
        
            
    def find_wave_index(self, wave, wave_val, perfect = True):
        """
        Returns a boolean array of indices such that wave[idx] matches wave_val entry
        
        Args:
            - wave(np.ndarray): List of all wavelengths
            - wave_val(np.ndarray): Wavelength values to find
            - perfect(bool, optional): If true, find only perfect matches, if false, find the closest match
        
        Returns:
            - np.ndarray: boolean array where true indicates a matching wavelength
        """
        
        if wave is None or len(wave) == 0:
            raise ValueError('Must define list of all wavelengths')
        
        if wave_val is None or len(wave_val) == 0:
            raise ValueError('Must define wavelength values')
        
        wave = np.array(wave)
        wave_val = np.array(wave_val)
        
        if perfect:
            idx = np.isin(wave, wave_val)
        else:
            idx = np.zeros(len(wave), dtype = bool)
            # For each wave_val, find the index in wave that is closest
            for val in wave_val:
                entry = np.argmin(np.abs(wave - val))
                idx[entry] = True
                
            # Check whether the same idx matched two wave_val entries
            n_found = np.sum(idx)
            if n_found != len(wave_val):
                print('Warning: Problems matching wavelengths. Could be out of range')
        
        return idx


if __name__ == '__main__':
    scene = sceneFromFile(input_data = r'eagle.jpg', 
                          image_type = 'RGB', 
                          wave_list = np.arange(400, 701, 10),
                          disp_cal = 'LCD-Apple.mat')

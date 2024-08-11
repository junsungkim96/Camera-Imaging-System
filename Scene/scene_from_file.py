import os
import numpy as np
import scipy.io as sio
import imageio
from skimage import img_as_float
from scipy.interpolate import interp1d


import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings


from Illuminant.illuminant import Illuminant
from Scene.scene import Scene
from Display.display import Display
from Utils.utils import energy_to_quanta, rgb_to_xw, xw_to_rgb


def sceneFromFile(input_data: str | None = None, 
                  image_type: str | None = None, 
                  mean_luminance: int | None = None, 
                  disp_cal = None,
                  wave_list = None, 
                  illuminant_energy = None, 
                  scale_reflectance = True):
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
            input_data = imageio.imread(input_data)

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
            elif isinstance(disp_cal, dict) and disp_cal.type == 'display':
                theDisplay = disp_cal
            else:
                raise ValueError("Bad display information.")

            # Set the display wavelength if wave_list is provided
            if wave_list is not None:
                theDisplay['wave'] = wave_list
            wave = theDisplay['wave']
            
            do_sub = False
            sz = None

            # Get the scene spectral radiance using the display model
            photons = read_image_file(input_data, image_type, theDisplay, do_sub, sz)

            ## Initialize scene
            # Match the display wavelength and the scene wavelength
            scene = Scene('rgb')
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
            il.energy = illuminant_energy

            # Compute photons for reflective display
            # For reflective display, until this step, the photon variable stores reflectance information
            if theDisplay.is_emissive:
                il_photons = il.photons
                il_photons = il_photons.reshape(1, 1, len(wave))
                photons = photons * il_photons

            # Set viewing distance and field of view
            scene.distance = theDisplay.distance
            img_size = photons.shape[1]
            img_fov = img_size * theDisplay.deg_per_dot
            scene.wangular = img_fov
            scene.distance = theDisplay.viewing_distance
            
        case 'spectral' | 'multispectral' | 'hyperspectral':
            if not isinstance(input_data, str):
                raise ValueError("Name of existing file required for multispectral")
            if wave_list is None:
                wave_list = []

            scene = Scene('multispectral')

            # Read image and illuminant structure
            photons, il, basis = read_image_file(input_data, image_type, wave_list)

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
            illuminant_energy =scene.illuminant_energy
            scene.illuminantenergy = illuminant_energy * max_reflectance

    return scene


def read_image_file(fullname, image_type = 'rgb', *args):
    
    # These are loaded for a file, when they are returned
    mc_coef = None
    comment = ''
    
    
    image_type = image_type.lower().replace(" ", "")
    
    match image_type:
        case 'rgb' | 'unispectral' | 'monochrome':
            disp_cal = args[0] if len(args) > 0 else None
            do_sub = args[1] if len(args) > 1 else False
            sz = args[2] if len(args) > 2 else None
            
            if isinstance(fullname, str):
                in_img = np.double(plt.imread(fullname))
            else:
                in_img = np.double(fullname)
                
            if disp_cal is None:
                if in_img.ndim == 2:
                    in_img = np.repeat(in_img[:, :, np.newaxis], 3, axis = 2)
                if in_img.ndim != 3:
                    raise ValueError(f'Bad number of dimensions {in_img.ndim} of image')
                
                print('Using block matrix primaries')
                xw_img = in_img.reshape((-1, 3)) / 255
                xw_img = np.clip(xw_img, 1e-3, 1)
                
                color_block_matrix = np.eye(3)
                photons = xw_img @ np.linalg.pinv(color_block_matrix[31])
            else:
                if isinstance(disp_cal, str):
                    the_display = Display(disp_cal)
                elif isinstance(disp_cal, dict) and disp_cal['type'] == 'display':
                    the_display = disp_cal
                else:
                    raise ValueError('Unknown display structure')
                
                wave = the_display['wave']
                g_table = the_display['gamma_table']
                n_primaries = the_display['n_primaries']
                
                if in_img.ndim == 2:
                    in_img = np.repeat(in_img[:, :, np.newaxis], n_primaries, axis = 2)
                    
                if in_img.shape[2] != n_primaries:
                    in_img = np.pad(in_img, ((0, 0), (0, 0), (0, n_primaries - in_img.shape[2])), 'constant')
                    
                if in_img.max() > len(g_table):
                    img_max = in_img.max()
                    n_bits = int(np.floor(np.log2(img_max)))
                    warnings.warn(f'Input data ({n_bits}) exceeds gTable ({int(np.log2(len(g_table)))}bits)')
                    max_display = 2 ** (n_bits + 1)
                    x_in = np.linspace(0, 1, len(g_table))
                    y_in = g_table
                    x_out = np.linspace(0, 1, 2 ** (n_bits + 1))
                    g_table = interp1d(x_in, y_in, kind = 'spline')(x_out)
                    in_img = in_img / img_max * (len(g_table) - 1)
                elif in_img.max() <= 1:
                    in_img = np.round(in_img * (len(g_table) - 1))
                elif in_img.max() <= 255:
                    s = len(g_table)
                    if s > 256: 
                        in_img = np.round(in_img / 255 * (s - 1))
                        
                in_img = lut_digital(in_img, 255 * (s - 1))
                
                if do_sub:
                    in_img = the_display.compute(in_img, sz)
                    
                spd = the_display['spd']
                xw_img = in_img.rehspae((-1, in_img.shape[2]))
                
                if xw_img.size < 1e-6:  # Use threshold as needed
                    photons = energy_to_quanta(wave, xw_img * spd.T)
                else:
                    photons = np.zeros((xw_img.shape[0], len(wave)))
                    for i in range(len(wave)):
                        photons[:, i] = energy_to_quanta(wave[i], xw_img @ spd[i, :].T)
                        
            photons = photons.reshape((in_img.shape[0], in_img.shape[1], -1))
            
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
                    new_bases = np.zeros((len(new_wave), basis['basis'].shape[1]))
                    for i in range(basis['basis'].shape[1]):
                        new_bases[:, i] = np.interp(new_wave, old_wave.flatten(), basis['basis'][:, i], \
                                                    left = extrap_val, right = extrap_val)
                    basis['basis'] = new_bases
                    basis['wave'] = new_wave

                photons = image_linear_transform(mc_coef, basis['basis'].T)
                if 'imgMean' in variables:
                    img_mean = variables['imgMean']
                    if len(args) > 0 and args[0] is not None:
                        extrap_val = 0
                        img_mean = np.interp(new_wave, old_wave.flatten(), img_mean.flatten(),
                                             left = extrap_val, right = extrap_val)
                    photons += img_mean.flatten()
                    
                illuminant = variables.get('illuminant', None)
                if illuminant is None:
                    warnings.warn(f'No illuminant information in {fullname}')
                
                photons = np.maximum(photons, 0)
            else:
                photons = variables.get('photons', variables.get('data', None))
                if photons is None:
                    raise ValueError('No photon data in file')
                
                comment = variables.get('comment', '')
                wave = variables.get('wave', variables.get('wavelength', None))
                if wave is None:
                    raise ValueError('No wavelength data in file')
                
                if len(args) > 0 and args[0] is not None:
                    new_wave = args[0]
                    idx = find_wave_index(wave, new_wave)
                    photons = photons[:, :, idx]
                    wave = new_wave
                
                basis = {'basis': [], 'wave': np.round(wave)}
                
                illumiant = variables.get('illuminant', None)
                if illuminant is None:
                    warnings.warn(f'No illumiannt information in {fullname}')
                
                if len(args) > 0 and args[0] is not None:
                    illuminant.wave = new_wave
        case _:
            raise ValueError(f'Unknown image type: {image_type}')
        
    return photons, illuminant, basis, comment, mc_coef


def image_linear_transform(image, transform):
    """
    Apply a linear transformation to the channels of an RGB or XW image
    
    Args:
        - 
    
    Returns:
        - 
    """
    # Determine image format to convert to XW
    if image.ndim == 3 or (image.ndim == 2 and len(transform) == 1):
        format = 'RGB'
        [r, c, w] = image.shape
        
        image = rgb_to_xw(image)
    else:
        format = 'XW'
        [_, w] = image.shape
    
    if len(transform) != w:
        raise ValueError('image/transform data sizes mismatch')
    
    # Multiply and reformat back to RGB if necessary
    image_transform = image @ transform
    
    if format == 'RGB':
        image_transform = xw_to_rgb(image_transform, r, c)
    
    return image_transform
    
        
def find_wave_index(wave, wave_val, perfect = True):
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


def lut_digital(DAC, g_table = None):
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
        
    if len(g_table) == 1:
        RGB = DAC ** g_table
        return RGB
    
    if np.max(DAC) > g_table.shape[0]:
        raise ValueError(f'Max DAC value {np.max(DAC)} exceeds the row dimension {g_table.shape[0]} of the g_table')
    
    if np.max(g_table) > 1 or np.min(g_table) < 0:
        raise ValueError(f'g_table entries should be between 0 and 1.')
    
    # Convert through the table
    RGB = np.zeros(DAC.shape)
    g_table = np.repeat(g_table[:, np.newaxis], DAC.shape[2], axis = 1)
    
    for i in range(DAC.shape[2]):
        this_table = g_table[:, i]
        RGB[:, :, i] = this_table[DAC[:, :, i]]
        
    return RGB


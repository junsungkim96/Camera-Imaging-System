# Python built-in modules
from warnings import warn
from pathlib import Path

# Pip-installed packages
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio

def get_image_format(data, wave):
    """
    Determine the ISET image format, either RGB or XW, from data.

    Args:
        data (np.ndarray): Image data
        wave (np.ndarray): Wavelength samples

    Returns:
        str: Image format ('RGB' or 'XW')

    Examples:
        data = np.random.rand(10, 10, 31)
        wave = np.arange(400, 701, 10)
        get_image_format(data, wave)
    """
    
    # A matrix in RGB format and the 3rd dimension equals the length of wave
    if data.ndim == 3 and data.shape[2] == wave.shape[-1]:
        return 'RGB'

    # A matrix in RGB format but with only one wavelength
    elif data.ndim == 2 and len(wave) == 1:
        return 'RGB'

    # A matrix in XW format and the 2nd dimension equals the length of wave
    elif data.ndim == 2 and data.shape[1] == wave.shape[-1]:
        return 'XW'

    # XW format for one point.
    # The data is a vector with the same number of entries as wave
    elif len(data) == wave.shape[-1]:
        return 'XW'

    else:
        raise ValueError("Unrecognized image format. Data shape: {}, wave length: {}".format(data.shape, len(wave)))

def quanta_to_energy(wavelength, photons):
    """
    Convert quanta (photons) to energy (watts)

    Args:
        wavelength: A vector describing the wavelength samples in nanometers (nm)
        photons: The matrix representing the photons in either RGB or XW (space-wavelength) format. Caution to the order

    Returns:
        numpy.ndarray: The energy (watts or joules) represented in the same format as the input (RGB or XW).

    The input matrix 'photons' can be in either RGB or XW format. In the XW format, each spatial position
    is in a row and the wavelength varies across columns. The output 'energy' [watts or joules] is returned
    in the same format as the input (RGB or XW) 
    """

    # Fundamental constants
    h = 6.62607015e-34  # Planck's constant [J sec]
    c = 299792458  # Speed of light [m/sec]
    
    if photons.size == 0:
        return np.array([])  # Return empty array if photons is empty

    wavelength = wavelength.reshape(1, -1)  # Make wavelength a row vector

    # Determine if the 'photons' format is in 'RGB' or 'XW
    format = get_image_format(photons, wavelength)

    if format == 'RGB':
        _, _, w = photons.shape
        if w != wavelength.shape[-1]:
            raise ValueError(
                'quanta_to_energy: photons third dimension must be nWave')
        energy = (h*c / 1e-9) * (photons / wavelength)
        
    elif format == 'XW':
        if photons.ndim == 1:
            # If photons is a vector, it must be a row
            photons = photons.reshape(1, -1)
        if photons.shape[1] != len(wavelength):
            raise ValueError('photons (quanta) must be in XW format')
        energy = (h*c / 1e-9) * (photons / wavelength)

    else:
        raise ValueError('Unknown image format')

    return energy

def energy_to_quanta(wavelength, energy):
    """
    Convert energy (watts) to number of photons

    Args:
        wavelength: The vector of sample wavelengths in units of nanometers(nm).
        energy: The matrix representing energy as a function of wavelength. If energy is in 2D it must be in space-energy format. Caution to the order

    Returns:
        photons: The number of photons in units of photons/sr/m^2/nm

    Examples:
        waves = np.arange(400, 701, 10)
        in_ = np.concatenate([blackbody(wave, 5000, 'energy'), blackbody(wave, 6000, 'energy')], axis = 1)
        p = energy_to_quanta(wave, in_) 
        plt.plot(wave, p)

    Notice that in the return, out becomes a row vector, consistent with XW format. 
    Also notice that we have to transpose p to make this work.

    """

    # Fundamental constants
    h = 6.62607015e-34  # Planck's constant [J sec]
    c = 299792458  # Speed of light [m/sec]

    # Wavelength must be a row vector
    wavelength = wavelength.reshape(1, -1) 

    if energy.ndim == 3:
        n, m, w = energy.shape
        if w != wavelength.shape[-1]:
            raise ValueError('energy_to_quanta: energy must have third dimension length equal to numWave')
        energy = np.reshape(energy, (n*m, w))
        photons = energy * (1e-9 * wavelength) / (h*c) 
        photons = np.reshape(photons, (n, m, w))
    elif energy.ndim == 2:
        n, w = energy.shape
        if w != wavelength.shape[-1]:
            raise ValueError('energy_to_quanta: energy must have a column length equal to numWave')
        photons = energy * (1e-9 * wavelength) / (h*c)
    elif energy.ndim == 1:
        energy = energy.reshape(1, -1)
        n, w = energy.shape
        if w != wavelength.shape[-1]:
            raise ValueError('energy_to_quanta: energy must have a column length equal to numWave')
        photons = energy * (1e-9 * wavelength) / (h*c)
        
    return photons

def xw_to_rgb(imXW, row, col):
    """
    Convert XW format data to RGB format

    Args:
        imXW: Data in XW format
        row: # of rows
        col: # of cols

    Returns:
        Data in RGB format

    This routine converts from XW format to RGB format. The row and column of the imXW are requried input
    arguments.

    We say matrices in (r, c, w) format are in RGB format. The dimension, w represents the number of data
    color bands. When w = 3, the data are an RGB image. But w can be almost anything (e.g., 31 wavelength
    samples from 400:10:700). We use this format frequently for spectral data

    The RGB format is useful for imaging. When w = 3, you can use conventional image() routines.

    The XW (space-wavelength) format is useful for computation. In this format, for example, 
    XW * spectralFunction yields a spectral response.

    The inverse is rgb_to_xw
    
    """

    if imXW is None or imXW.size == 0:
        raise ValueError('No image data.')

    if row is None or col is None:
        raise ValueError('No row or col size.')

    x = imXW.shape[0]
    w = imXW.shape[1]

    if row * col != x:
        raise ValueError('xw_to_rgb: Bad row, col values')

    imRGB = imXW.reshape((row, col, w), order = 'F')

    return imRGB

def rgb_to_xw(imRGB):
    """
    Transform RGB form matrix into XW (space-wavelength) matrix

    Args:
        imRGB: Input matrix in RGB format

    Returns:
        XW: Matrix in XW format
        r: # of rows
        c: # of columns
        w: # of color bands (wavelengths)
    """
    s = imRGB.shape

    if len(s) < 3:
        s += (1, )

    if len(s) < 4:
        XW = imRGB.reshape((s[0] * s[1], s[2]), order = 'F')
    else:
        XW = imRGB.reshape((s[0] * s[1], s[2], s[3]), order = 'F')

    r, c, w = s[:3]

    return XW, r, c, w

def blackbody(wave, temps, unitType='energy', eqWave=550):
    """
    Generate the spectral power distribution of a blackbody radiator

    Args:
        wave (np.ndarray): Wavelength samples
        temps (np.ndarray): Array of color temeprature in degress Kelvin
        unitType (str): Type of units to return ('energy' or 'photons')
        eqWave (float): Wavelength at which the radiance is scaled.

    Returns:
        specRad (np.ndarray): Spectral power distribution
        XYZ (np.ndarray): XYZ color values if 'unitType' is 'photons' or 'energy'.
    """
    # Fundamental constants
    c1 = 3.741832e-16  # [W m^2]
    c2 = 1.438786e-2  # [m K]

    # Convert wavelengths from nm to m
    waveM = np.array(wave) * 1e-9

    # Calculate spectral radiance using Planck's law
    specEmit = c1 / (waveM[:, np.newaxis]**5 *
                     (np.exp(c2 / waveM[:, np.newaxis] * temps)) - 1)
    specRad = specEmit * 1e-9 / np.pi  # [W/(m^2 nm sr)]

    # Normalize to match radiance at eqWave and adjust for luminance
    idx = np.argmin(np.abs(np.array(wave) - eqWave))
    s = specRad[idx, 0] / specRad[idx, :]
    specRad *= s
    L = np.mean(luminance_from_energy(specRad[:, 0], wave))
    specRad *= (100 / L)

    # Convert to photons units if requested
    if unitType.lower() == 'photons':
        specRad = energy_to_quanta(wave, specRad)

    return specRad

def xyz_from_photons(photons, wave):
    """
    Convert photon spectral power distribution into CIE XYZ.

    Args:
        photons (np.ndarray): Spectral power distribution in phtons
        wave (np.ndarray): Wavelength samples

    Returns:
        XYZ (np.ndarray): CIE XYZ values
    """
    energy = quanta_to_energy(wave, photons)
    xyz = xyz_from_energy(energy, wave)

    return xyz

def luminance_from_energy(energy: np.ndarray, 
                          wave: np.ndarray,
                          bin_width: int | None = None) -> float:
    """
    Calculate luminance(cd/m2) and related quantities(lumen, cd, lux) from spectral energy.


    The CIE formula for luminance converts a spectral radiance distribution(W/m2-sr-nm) into
    luminance(candelas per meter squared, cd/m2).
    This routine accepts RGB(3d) or XW(space-wavelength)(2d) formatted inputs.
    In XW format, the spectral distributions are in the rows of the ENERGY matrix.
    (Still can accept column vectors)


    The formula for luminance and illuminance are the same, differing only in the units of the input.
    Hence, this routine calculates illuminance(lux) from a spectral irradiance distribution(W/m2 nm).
    It also calculates luminous intensity(cd) from spectral radiant intensity(W/sr nm);
    finally, it calculates luminous flux(lumen, lm) from spectral power(W/nm).
    The pairings are:
        - Luminance: cd/m2 from spectral radiance(W/m2-sr-nm)
        - Illuminance: lux(lm/m2) from spectral irradiance(W/m2-nm)
        - Luminous flux: lumen(cd-sr) from spectral power(W/nm)
        - Luminous intensity: cd from spectral radiant intensity(W/sr-nm)

    Args:
        energy: spectral radiance(watts/sr/nm/m2)  (a vector, or a matrix XW format)
        wave: wavelength samples (a column vector)
        bin_width: wavelength bin width when there is a monochromatic input, default is None

    Returns:
        lum: Luminance in cd/m2

    References:
        https://en.wikipedia.org/wiki/Luminous_efficiency_function

        https://wp.optics.arizona.edu/jpalmer/radiometry/radiometry-and-photometry-faq/

    """
    if bin_width is None:
        bin_width = wave[1] - wave[0]

    match get_image_format(energy, wave):
        case 'RGB':  # 3d
            xw_data, *_ = rgb_to_xw(energy)

        case 'XW':  # 2d or 1d
            xw_data = energy

        case _:
            raise ValueError('Energy must be 2D or 3D')

    V = read_spectral('data/human/luminosity.mat', wave)

    # The luminance formula
    # 683 is a constant that converts the spectral radiance to luminance
    lum = 683 * xw_data @ V * bin_width

    return lum

def unit_conversion(units):
    """
    Return scale factor that converts from meters or seconds to other scales
    
    Args:
        units(str): Name of the units
        
    Returns:
        float: scale factor
    """
    
    units = units.lower()
    
    # Convert space
    if units in ['nm', 'nanometer', 'nanometers']:
        return 1e9
    
    elif units in ['micron', 'micrometer', 'um', 'microns']:
        return 1e6
    
    elif units in ['mm', 'millimeter', 'millimeters']:
        return 1e3
    
    elif units in ['cm', 'centimeter', 'centimeters']:
        return 1e2
    
    elif units in ['m', 'meter', 'meters']:
        return 1
    
    elif units in ['km', 'kilometer', 'kilometers']:
        return 1e-3
    
    elif units in ['inches', 'inch']:
        return 39.37007874  # inches/meter
    
    elif units in ['foot', 'feet']:
        return 3.280839895  # feet/meter
    
    # Convert time
    elif units in ['s', 'second', 'sec']:
        return 1
    
    elif units == 'ms':
        return 1e3
    
    elif units == 'us':
        return 1e6
    
    # Convert radian angles    
    elif units in ['degrees', 'deg']:
        return 180 / np.pi
    
    elif units == 'arcmin':
        return (180 / np.pi) * 60
    
    elif units == 'arcsec':
        return (180 / np.pi) * 60 * 60
    
    else:
        raise ValueError('Unknown spatial units specification')
    
def sample_to_space(r_samples, c_samples, row_delta, col_delta) -> tuple[np.ndarray, np.ndarray]:
    """
    The physical position of samples in a scene or optical image.

    Given a number of row and column samples(r_samples, c_samples) and a spacing between the rows and columns
    (row_delta, col_delta), return the full sampling grid in the units of row_delta and col_delta.
    These are usually in units of micron for these sensor applications.

    We treat the center of the samples as (0, 0) and use the sampling spacing in microns to calculate the location of
    the other samples.

    Args:
        r_samples: Number of row samples
        c_samples: Number of column samples
        row_delta: distance between rows
        col_delta: distance between columns

    Returns:
        r_microns:
        c_microns:
    """

    r_center = np.mean(r_samples)
    c_center = np.mean(c_samples)

    r_microns = (r_samples - r_center) * row_delta
    c_microns = (c_samples - c_center) * col_delta

    return r_microns, c_microns

def lrgb2srgb(rgb: np.ndarray) -> np.ndarray:
    """
    Args:
        rgb:

    Returns:
    """

    assert rgb.max() <= 1. and rgb.min() >= 0., 'Linear image must be in the range [0, 1]'

    big = rgb > 0.0031308
    rgb[~big] = rgb[~big] * 12.92
    rgb[big] = 1.055 * np.power(rgb[big], 1 / 2.4) - 0.055

    return rgb

def read_color_filter(wave: np.ndarray, f_name: str | Path) -> tuple[np.ndarray, str | None]:
    """
    Read transmission data and name for a set of color filters.

    The color filter information includes the wavelength transmission curve and color filter names.
    The filter curves are a spectral case of reading spectral data of two reasons.
        1. The use of filter name,
        2. the requirement that the transmission values run between 0 and 1.
    These requirements lead to a special functions, different from the more commonly used function, `read_spectral`.

    If values exceed 1, the data are scaled. If the values below 0, the returned variables are set to null.

    In some cased, users add additional data to the file for
    Args:
        wave: The wavelength samples
        f_name: The file name or path to load the filter data

    Returns:
        data: The filter data
        filter_names: The filter names
    """

    assert wave.ndim == 1, 'The wavelength samples must be a 1D array'

    data = read_spectral(f_name, wave)
    if data is None:
        raise ValueError(f'No data found in {f_name}')

    if np.max(data) > 1:
        warn(f'Maximum value in {f_name} exceeds 1.0, scaling to 1.0')
        data = scale(data, 1)

    if np.min(data) < 0:
        warn(f'Minimum value in {f_name} is less than 0.0, check the data')

    """
    If there are filter names, return them. This is a cell array describing the each column of data. 
    The first character is from the list in sensor_color_order, rgbcmyw. We could/should check (enforce) this here.
    """
    file_data = loadmat(f_name)
    try:
        filter_names = file_data['filterNames']
    except KeyError:
        filter_names = None

    return data, filter_names

def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
                elem.size > 0) and isinstance(
            elem[0], sio.matlab.mio5_params.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          sio.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          sio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          sio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = sio.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def read_spectral(file_path: str | Path, 
                  wave: np.ndarray | None = None, 
                  extrap_val: float | str = 0.,
                  make_positive: bool = False) -> np.ndarray:
    """
    Read in spectral data and interpolate to the specified wavelengths


    Spectral data are stored in files that include both the sampled data and the wavelength values.
    This routine reads the stored values and returns them interpolated or extrapolated to the values in parameter
    wave.

    Args:
        file_path: file name to read.
        wave: Wavelength samples.
        extrap_val: Extrapolation values for wavelengths outside the range in the file. default is 0.0.
        make_positive: Make sure the first basis in mainly positive.

    Returns:
        res: Interpolated spectral data.
    """
    
    file_path = str(file_path)
    tmp = loadmat(file_path)
    data = tmp['data']
    wavelength = tmp['wavelength']
    comment = tmp['comment']  # in the future, we may use comment to print out the information of the data
    
    # the data's ndim load from the mat file is 2.
    assert data.shape[0] == wavelength.size, f'Mis-match between wavelength and data in {Path(file_path).name}'

    if wave is None:
        wave = wavelength.flatten()

    """
    If the data is a single column, then we can interpolate it directly
    Otherwise, we need to loop over each column, because interp1d only works on 1D arrays.
    We assume data is a 2D array, if last dimension is 1, then we treat it as a single column.
    else, we loop over the columns.
    
    maybe we can integrate the two cases into one.
    """
    if data.ndim == 1:
        data = data[..., np.newaxis]
    
    if data.shape[-1] == 1:
        # ravel return contiguous view of the origin array
        # flatten return a copy of the origin array
        f = interp1d(wavelength.ravel(), data.ravel(), kind='linear', fill_value=extrap_val, bounds_error=False)
        # Use this function to interpolate the data at the points specified by wave
        res = f(wave)
        # Maybe we can use numpy.interp to do the same thing. need to check...

    else:
        # Initialize an empty array for the results
        res = np.zeros((len(wave), data.shape[1]))

        # Loop over each column in data
        for i in range(data.shape[1]):
            # Create interpolation function for this column
            f = interp1d(wavelength.ravel(), data[:, i], kind='linear', fill_value=extrap_val, bounds_error=False)
            # Interpolate the data at the points specified by wave and store in res
            res[:, i] = f(wave)

    if make_positive and np.mean(res[:, 0]) < 0:
        res *= -1

    return res

def scale(im: np.ndarray, b1: float | None = None, b2: float | None = None):
    """
    Scale the values in im into the specified range.

    Changes in syntax produce different scaling operations:
    im = scale(im)                             scale from 0 to 1
    im = scale(im, b1, b2)                     scale and offset to range b1, b2
    im = scale(im, max_value)                  scale largest value to max_value

    Args:
        im (np.ndarray): input data
        b1 (float, optional): lower bound of the range. Defaults to None.
        b2 (float, optional): upper bound of the range. Defaults to None.

    Returns:
        np.ndarray: scaled data
    """
    # Find data range
    mx = np.max(im)
    mn = np.min(im)

    # If only one bounds argument, just set peak value
    if b1 is not None and b2 is None:
        im = im * (b1 / mx)
        return im

    # If 0 or 2 bounds arguments, we first scale data to 0,1
    im = (im - mn) / (mx - mn)

    if b1 is None and b2 is None:
        # No bounds arguments, assume 0,1
        b1 = 0
        b2 = 1
    elif b1 is not None and b2 is not None:
        if b1 >= b2:
            raise ValueError('scale: bad bounds values.')

    # Put the (0,1) data into the range
    range_ = b2 - b1
    im = range_ * im + b1

    return im

def xyz_from_energy(energy: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    # Energy is in RGB format
    r, c, _ = energy.shape
    
    S = read_spectral('data/human/XYZ.mat', wavelength)
    
    binwidth = wavelength[1] - wavelength[0]

    # Change the energy format from RGB to XW
    energy = energy.reshape(r*c, -1)
    
    XYZ = 683 * (energy @ S) * binwidth
    
    # Change the XYZ format from XW to RGB
    XYZ = XYZ.reshape(r, c, -1)
    
    # Divide by the maximum value for normalization
    XYZ /= np.max(XYZ)

    return XYZ

# def xyz_from_energy(energy, wave):
#     """
#     CIE XYZ values from spectral radiance (W/nm/sr/m^2) or irradiance (W/nm/m^2)

#     Args:
#         energy(np.ndarray): Spectral radiance or irradiance functions
#         wave(np.ndarray): Wavelength samples

#     Returns:
#         XYZ(np.ndarray): XYZ values (X, Y, Z) in XW format
#     """
#     # Force data into XW format
#     if energy.ndim == 3:
#         if len(wave) != energy.shape[2]:
#             raise ValueError('Bad format for input variable energy.')

#     # Check the input format
#     iFormat = get_image_format(energy, wave)
#     if iFormat == 'RGB':
#         # Convert to XW format
#         xwData, r, c = rgb_to_xw(energy)
#     else:
#         # XW format
#         xwData = energy

#     # Read XYZ color matching functions
#     S = read_spectral('data/human/XYZ', wave)
#     dWave = wave[1] - wave[0] if len(wave) > 1 else 10

#     # Calculate XYZ values
#     xyz = 683 * np.dot(xwData, S) * dWave

#     # If input was in RGB format, return in RGB format
#     if iFormat == 'RGB':
#         xyz = xw_to_rgb(xyz, r, c)

#     return xyz
# Python built-in modules
import copy
from pathlib import Path
from warnings import warn
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pip-installed packages
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.constants import elementary_charge

# Local modules
from Illuminant.illuminant import Illuminant
from Scene.scene import Scene
from Optics.optics import OI
from Sensor.pixel import Pixel
from Utils.utils import lrgb2srgb, read_color_filter, sample_to_space, rgb_to_xw, xw_to_rgb


q = 1.602177e-19  # Elementary charge, C

class Sensor():
    """
    The sensor array uses a pixel definition that can be specified in the parameter Pixel.
    If this is not passed in, then a warning will be issued and the default pixel is created and returned.

    Several type of image sensors can be created, including multi-spectral and a model of the human cone mosaic.

    Bayer RGB combinations:
        1. bayer-grbg
        2. bayer-rggb
        3. bayer-bggr
        4. bayer-gbrg

    Bayer CMY combinations:
        1. bayer-ycmy
        2. bayer-cyym

    Vendor parts calibrated over the years or from the web:
        1. MT9V024: Omnivision
        2. ar0132at: An ON sensor used in automotive applications
        3. imx363: A widely used Sony digital camera sensor (used in the Google Pixel 4a)
        4. imx490-large: The Sony imx490 sensor large
        5. imx490-small: The Sony imx490 sensor small
        6. nikon d100: An older model Nikon (D100)

    Other types:
        1. monochrome: A Single monochrome sensor
        2. monochrome array: cell array of monochrome sensors
        4. lightfield: RGB to match the resolution of a lightfield oi
        5. dualpixel: RGB dual pixel for autofocus (Bayer)

    Multiple channel sensors can be created
        1. grbc: green, red, blue, cyan
        2. rgbw: One transparent channel
        3. imec44: IMEC 16 channel sensor, 5.5um
        4. custom

    Human cone mosaic:
        1. human: Uses Stockman Quanta LMS cones.

    """

    def __init__(self,
                 sensor_type: str = 'default',
                 *,
                 pixel: Pixel | None = None,
                 **kwargs):
        
        self.sensor_type = sensor_type.lower().replace(" ", "")
        self.name = None
        self.rows: int | None = None
        self.cols: int | None = None
        self.oi = None
        
        if pixel is None:
            warn('No pixel definition passed in. Using default pixel.')
            pixel = Pixel('default')
            self.pixel = pixel
            self.size = self.get_size('qqicf')  # set the default row/col size
        # TODO: Implement more Pixel types
        else:
            self.pixel = pixel

        self.spectrum: dict = {'wave': self.pixel.spectrum}
        self.data = None

        # Initialize the sensor parameters
        self._sigma_gain_fpn: float | None = None
        self.gain_fpn_image: np.ndarray | None = None  # prnu_image
        # self.sigma_gain_fpn = 0  # [V/A] This is the slop of the transduction function

        self._sigma_offset_fpn: float | None = None
        self.offset_fpn_image: np.ndarray | None = None  # dsnu_image
        # self.sigma_offset_fpn = 0  # [V] This is the offset from 0 volts after reset

        self.analog_gain = 1
        self.analog_offset = 0
        # below two codes are duplicated with sigma_offset_fpn and sigma_gain_fpn, but we still keep them for now
        self.offset_fpn_image = None
        self.gain_fpn_image = None
        self.quantization = self.get_quantization('analog')

        self.cfa_pattern: np.ndarray | None = None
        self.filter_file: str | None = None
        self.color: dict | None = None

        self.n_samples_per_pixel: int = 1  # Number of samples per pixel (default is 1)

        match self.sensor_type:
            case 'default' | 'color' | 'bayer' | 'rgb' | 'bayer-grbg':
                filter_order = np.array([[2, 1],
                                         [3, 2]])
                filter_file = 'RGB'
                self.create_sensor_bayer(filter_order, filter_file)

            case 'bayer-rggb':
                filter_order = np.array([[1, 2],
                                         [2, 3]])
                filter_file = 'RGB'
                self.create_sensor_bayer(filter_order, filter_file)
                
            case 'bayer-bggr':
                filter_order = np.array([[3, 2],
                                         [2, 1]])
                filter_file = 'RGB'
                self.create_sensor_bayer(filter_order, filter_file)
            
            case 'bayer-gbrg':
                filter_order = np.array([[2, 3],
                                         [1, 2]])
                filter_file = 'RGB'
                self.create_sensor_bayer(filter_order, filter_file)
            
            case 'bayer-ycmy':
                filter_order = np.array([[2, 1],
                                         [3, 2]])
                filter_file = 'cym'
                self.create_sensor_bayer(filter_order, filter_file)
            
            case 'bayer-cyym':
                filter_order = np.array([[1, 2],
                                         [2, 3]])
                filter_file = 'cym'
                self.create_sensor_bayer(filter_order, filter_file)
            
            case 'mt9v024':
                raise NotImplementedError('Sensor not implemented')
            
            case 'ar0132at':
                raise NotImplementedError('Sensor not implemented')
            
            case 'imx363':
                raise NotImplementedError('Sensor not implemented')
            
            case 'imx490-large':
                raise NotImplementedError('Sensor not implemented')
            
            case 'imx490-small':
                raise NotImplementedError('Sensor not implemented')
            
            case 'nikon-d100':
                raise NotImplementedError('Sensor not implemented')
            
            case 'ideal':
                raise NotImplementedError('Ideal sensor not implemented')
                
            case 'lightfield':
                raise NotImplementedError('Lightfield sensor not implemented')

            case 'dualpixel':
                raise NotImplementedError('Dual pixel sensor not implemented')

            # TODO: There are tons of sensor types to implement...
            case _:
                raise ValueError(f'Unknown sensor type {self.sensor_type}...')

        # Set the exposure time - this needs a CFA to be established to account for CFA exposure mode.from
        self.integration_time: float = 0
        self.auto_exposure: bool = True
        self.CDS = 0

        # Place holder for charts, such as the MCC
        self.chart_parameters = None
        # Compute with all noise turned on by default
        self.noise_flag = 2

        # Initialize for other attributes
        self.etendue: np.ndarray | None = None

        self.response_type: str = 'linear'

    def create_sensor_bayer(self, filter_pattern: np.ndarray, filter_file: str = 'RGB') -> None:
        """
        Create a default image sensor array.
        Args:
            filter_pattern: The filter pattern
            filter_file: The filter file to use

        Returns:
            None
        """
        self.name = f'Bayer'
        self.cfa_pattern = filter_pattern
        self.filter_file = filter_file
        self.color = dict()
        self.color['filter_spectra'], self.color['filter_names'] = self.sensor_read_color_filter()

    def sensor_read_color_filter(self):
        """
        Read the color filter array from a file.
        Returns:
            filter_spectra: The filter spectra
            filter_names: The filter names
        """
        wave = self.wave
        n_wave = len(wave)
        folder_path = Path(__file__).parent.parent / 'data'
        
        match self.filter_file.lower():
            case 'xyz':
                f_name = folder_path / 'human' / 'XYZ.mat'
            case 'rgb':
                f_name = folder_path / 'sensor/colorfilters' / 'RGB.mat'
            case 'cym':
                f_name = folder_path / 'sensor/colorfilters' / 'cym.mat'
            case 'grbc':
                f_name = folder_path / 'sensor/colorfilters' / 'GRBC.mat'
            case 'stockmanabs':
                f_name = folder_path / 'human' / 'stockmanabs.mat'
            case 'monochrome':
                filter_spectral = np.ones((n_wave, 1))
                file_names = 'w'
                return filter_spectral, file_names
            case _:
                raise ValueError(f'Unknown filter file {self.filter_file}...')
        
        filter_spectral, filter_names = read_color_filter(self.wave, f_name)
        return filter_spectral, filter_names

    @staticmethod
    def get_size(format_type: str):
        """
        Row/col size and sensor imaging area sizes of various formats.

        The row/col formats specified are qcif, cif, qvga, vga, svga, xvga, uxvga.
        Args:
            format_type: The format type

        Returns:
            dict: The row/col size

        """

        tmp = {"qqicf": (72, 88),
               'qcif': (144, 176),
               'qqvga': (120, 160),
               'qvga': (240, 320),
               'cif': (288, 352),
               'vga': (480, 640),
               'svga': (600, 800),
               'xvga': (768, 1024),
               'uvga': (1024, 1280),
               'uxvga': (1200, 1600),  # Not sure about this one.
               }

        tmp.update({'halfinch': (0.0048, 0.0064)})
        tmp.update({'quarterinch': (0.0024, 0.0032)})
        tmp.update({'sixteenthinch': (0.0012, 0.0016)})

        return tmp[format_type]

    @property
    def size(self):
        return self.rows, self.cols

    @size.setter
    def size(self, value: tuple[int, int]):
        self.rows, self.cols = value

        # TODO: Implement the rest of the setter
        # But it seems that rest of code us for `human` sensor type which is not implemented yet, so skipping it for now

    @property
    def sigma_offset_fpn(self):
        return self._sigma_offset_fpn

    @sigma_offset_fpn.setter
    def sigma_offset_fpn(self, value: float):
        self._sigma_offset_fpn = value
        # Clear the dsnu image when the dsnu level is reset
        self.offset_fpn_image = None

    @property
    def sigma_gain_fpn(self):
        return self._sigma_gain_fpn

    @sigma_gain_fpn.setter
    def sigma_gain_fpn(self, value: float):
        self._sigma_gain_fpn = value
        # Clear the prnu image when the prnu level is reset
        self.gain_fpn_image = None  # prnu_image

    @property
    def spectral_QE(self) -> np.ndarray:
        """
        Compute the sensor spectral QE

        Combine the pixel detector, the sensor color filters, and the infrared filter into a sensor spectral QE.
        Returns:
            spectral_QE: The sensor spectral QE
        """

    def get_quantization(self, q_method: str = 'analog') -> dict:
        """
        Set the quantization method and bits count for an image sensor array
        Args:
            q_method: The quantization method

        Returns:

        """
        quantization = dict()
        match q_method:
            case 'analog':
                quantization['bits'] = None
                quantization['method'] = 'analog'
            case '4bit':
                quantization['bits'] = 4
                quantization['method'] = 'linear'
            case '8bit':
                quantization['bits'] = 8
                quantization['method'] = 'linear'
            case '10bit':
                quantization['bits'] = 10
                quantization['method'] = 'linear'
            case '12bit':
                quantization['bits'] = 12
                quantization['method'] = 'linear'
            case 'sqrt':
                quantization['bits'] = 8
                quantization['method'] = 'sqrt'
            case 'log':
                quantization['bits'] = 8
                quantization['method'] = 'log'
            case _:
                raise ValueError(f'Unknown quantization method {q_method}...')

        return quantization

    @property
    def wave(self):
        return self.spectrum['wave']

    def pixel_center_fill_PD(self, fill_factor: float) -> None:
        """
        Adjust the pixel photodiode to be centered (fill_factor ~ (0, 1])

        Create a centered photodetector with a particular fill factor within a pixel.
        (The pixel is attached to the sensor.)

        Warnings:
            This function should probably be a part of the pixel class, not the isa.
            Also, changes in the size of the pixel within GUI preserve the fill factor.

        Args:
            fill_factor: fill factor of the pixel
        Returns:
            None
        """
        assert 0 < fill_factor <= 1, 'Fill factor must be between 0 and 1'

        pixel = self.pixel

        # TODO: Implement the rest of the function
        raise NotImplementedError('This function is not implemented yet')

    def compute(self, oi) -> None:
        """
        Compute the sensor response from the optical image instance.
        Get the computed voltage data.

        The computation checks a variety of parameters and flags in the sensor instance to perform the calculation.

        Noise flags:
        The noise flag in an important way to control the details of the calculation. The default value of the
        noise flag is 2. This case is standard operating mode with photon noise, read/reset, dsnu, prnu, analog gain
        /offset, clipping, quantization, included.

        Each of the noises can be individually controlled, but the noise flag simplifies turning off
        different types of noise for certain experiments
        The conditions are:

        | noiseFlag | photon | e-noises | PRNU/DSNU | clipping | CDS | Description |
        |-----------|--------|----------|-----------|----------|-----|-------------|
        | -2        | +      | 0        | 0         | 0        | 0   | no pixel no system |
        | -1        | 0      | 0        | 0         | 0        | 0   | no photon no pixel no system |
        | 0         | 0      | 0        | +         | +        | +   | no photon no pixel |
        | 1         | +      | 0        | +         | +        | +   | no pixel noise |
        | 2         | +      | +        | +         | +        | +   | default |


        photon noise:  Photon noise
        pixel noise:   Electrical noise (read, reset, dark) (e_noise)
        system noise:  gain/offset (prnu, dsnu), clipping, quantization (GCQ)

        - noiseFlag = -2 - +photon noise, -eNoise, -GCQ
        - noiseFlag = -1 - -photon noise, -eNoise, -GCQ
        - noiseFlag =  0 - -photon noise, -eNoise, +GCQ
        - noiseFlag =  1 - +photon noise, -eNoise, +GCQ
        - noiseFlag =  2 - +photon noise, +eNoise, +GCQ

        In addition to controlling factors through the noise flag, it is possible
        to manage them by individually setting sensor parameters. For example,
        when noiseFlag 0,1,2, you can still control the analog gain/offset noise
        can be eliminated by setting 'prnu sigma' and 'dsnu sigma' to 0.
        Similarly, you can set the read noise and dark voltage to 0.

        Quantization noise can be turned off by setting the quantization method
        to 'analog'

        * quantization       - set 'quantization method' to 'analog' (default)
        * CDS                - set the cds flag to false (default)
        * Clipping           - You can avoid clipping high with a large
            voltage swing.  But other noise factors might drive the
            voltage below 0, and we would clip.

        COMPUTATIONAL OUTLINE:

        1. Check exposure duration: autoExposure default, or use the set time.
        2. Compute the mean image: sensorComputeImage()
        3. Etendue calculation to account for pixel vignetting
        4. Noise, analog gain, clipping, quantization
        5. Correlated double-sampling
        6. Macbeth ROI management

        The value of showBar determines whether the waitbar is displayed to
        indicate progress during the computation.
        Args:
            oi: The optical image

        Returns:
            volts: The computed voltage data
        """
        # assert isinstance(oi, OpticalImage), 'The oi must be an instance of OpticalImage'
        self.oi = oi  # Store the oi for future computations
        # In matlab code, sensor may can have multiple sensor_array, but here we only have one sensor_array

        """
        Determine the exposure model - At this point, we use either the default auto-exposure or we use the time the 
        user set. If you would like to use a different exposure model, run it before the `compute` method and set the 
        integration time determined by the model. Some day, we might allow the user to set the model here, but 
        that is not currently the case.
        """
        integration_time: float = self.integration_time
        pattern = self.cfa_pattern
        if integration_time == 0 or self.auto_exposure:
            self.integration_time = self.get_auto_exposure_integration_time(0.95, 'default')
        else:
            # TODO: integtation_time is vector and same as pattern
            raise NotImplementedError('This part is not implemented yet')

        """
        Calculate current
        This factor converts pixels current to volts for this integration time. The conversion units are:
            sec * (V/e) * (e/charge) = V / (charge / sec) = V / current (A)
        
        Given the basic rule V = IR, k is effectively a measure of resistance that creating matching conversion from 
        current to volts.
        """
        q = elementary_charge
        pixel = self.pixel

        cur2volt = self.integration_time * pixel.conversion_gain / q

        unit_signal_current = self.signal_current(self.oi)
        # Convert to volts
        volt_image = cur2volt * unit_signal_current

        """
        Calculate etendue from pixel vignetting
        We want an (wavelength-independent) of scale factors that account for the loss of light at each pixel as a 
        function of the chief ray angle. This method only works for wavelength-independent relative illumination. 
        See notes in signal_current_density for another approach that we might use some day, say in ISET-3.0.
        
        """
        self.vignetting()
        etendue = self.etendue
        if self.n_exposures == 1:
            volt_image = volt_image * etendue
        else:
            volt_image = volt_image * np.tile(etendue, (1, 1, self.n_exposures))

        response_type = self.response_type
        match response_type:
            case 'log':
                raise NotImplementedError('This part is not implemented yet')
            case 'linear':
                pass
            case _:
                raise ValueError(f'Unknown response type {response_type}...')

        self.volts = volt_image

        # We have the mean image computed. Now add noise, clip and quantize.
        noise_flag = self.noise_flag
        ag = self.analog_gain
        ao = self.analog_offset

        if noise_flag >= 0:
            if noise_flag > 0:
                self.add_noise()

        # Add gain simulation
        if ag != 1 or ao != 0:
            raise NotImplementedError('This part is not implemented yet')


        # Clipping
        if 0 <= noise_flag <= 2:
            v_swing = pixel.voltage_swing
            self.volts = np.clip(volt_image, 0, v_swing)

        if noise_flag == -2:
            raise NotImplementedError('This part is not implemented yet')

        # Quantization
        # TODO: Write comments for quantization

        quantization_method = self.quantization['method']
        match quantization_method:
            case 'analog':
                pass
            case 'linear':
                raise NotImplementedError('This part is not implemented yet')
            case 'sqrt':
                raise NotImplementedError('This part is not implemented yet')
            case 'log':
                raise NotImplementedError('This part is not implemented yet')
            case 'gamma':
                raise NotImplementedError('This part is not implemented yet')
            case _:
                raise ValueError(f'Unknown quantization method {quantization_method}...')

        # Correlated double sampling

        if self.CDS and noise_flag >= 0:
            warn('CDS on')
            raise NotImplementedError('This part is not implemented yet')

        # Done!

    def get_auto_exposure_integration_time(self,
                                           level: float = 0.95,
                                           ae_method: str = 'default',
                                           **kwargs
                                           ) -> float:
        """
        Determine the exposure duration for the sensor.

        Find an integration time (sec) for the optical image and sensor. The default method produces a voltage level at
        a fraction (0< level < 1) of the voltage swing. The data used to set the level are from the signal current
        image plus the dark current image.

        These are additional exposure methods that can be specified using the ae_method parameter.

        The currently implemented methods are

        * 'default' - The default method finds the signal voltage for a one sec exposure of the portion of the
        optical image with peak illuminance. The exposure routine then returns an integration time that produces a
        value of "level" times the voltage swing. (ae_luminance)

        * 'luminance' - Same as default (ae_luminance)

        * 'full' - Finds the maximum signal voltage for a one sec exposure, and then returns an integration time
        that produces a value of 'level' times the voltage swing. (ae_full)

        * 'specular' -  Make the mean voltage level a fraction of voltage swing. (ae_specular)

        * cfa - Compute separately for each color filter type (ae_cfa)

        * mean - Finds the sensor integration time to achieve the desired mean voltage level expressed as a fraction
        of the voltage swing. (ae_mean). For example, setting level of 0.3 means that
            mean(volt_image(:)) = 0.3 * voltage_swing

        * weighted - Set the exposure duration as in 'luminance', but using a rect from the center of the image
        (ae_weighted).

        * video - Same as 'weighted', but a maximum exposure time sent in by a videomax parameter (default 1 / 60 s)
        (as_video)

        * hdr - Like specular, but clips from high and low to possibly provide a better baseline for hdr processing.

        Args:
            level: Fraction of the voltage swing, default is 0.95
            ae_method: The method of auto exposure
            kwargs: Additional keyword arguments
                centerrect (optional): Central rectangle for 'center' method


        Returns:
            integration_time: The exposure duration
        """
        # These parameters are not used in the `ae_luminance` method
        center_rect = kwargs.get('center_rect', None)
        videomax = kwargs.get('videomax', 0.95)
        num_frames = kwargs.get('num_frames', 1)

        match ae_method:
            case 'default' | 'luminance':
                integration_time = self.ae_luminance(level)

            # TODO: Implement the rest of the cases

            case _:
                raise ValueError(f'Unknown auto exposure method {ae_method}...')

        return integration_time

    def ae_luminance(self, level: float = 0.95) -> float:
        """
        The default auto exposure method.

        Extracts the brightest part of the image(`extract_bright`) and sets the integration time so that the brightest
        part is at a fraction of the voltage swing.

        Because this method only calculates the voltages for a small portion of the image (the brightest), it is a lot
        faster than computing the full image.
        We are not sure if in practice only the brightest part can be extracted.

        Args:
            level: The fraction of the voltage swing of the brightest part of the image
        Returns:
            integration_time: The exposure duration
        """

        voltage_swing = self.pixel.voltage_swing

        # Find the brightest pixel in the optical image.
        small_oi = self.oi.extract_bright()

        # The number of sensors must match the CFA format of the color sensor arrays. So we choose a size that is an
        # 8x multiple of the CFA pattern.
        small_sensor = copy.deepcopy(self)
        small_sensor.size = (8 * self.cfa_size[0], 8 * self.cfa_size[1])

        # We clear the data as well.
        small_sensor.clear_data()
        small_sensor.integration_time = 1
        # s_distance = small_sensor.oi.scene.distance
        # distance = small_sensor.oi.get_focal_plane_distance(s_distance)
        # width = small_sensor.cols * (small_sensor.pixel.width + small_sensor.pixel.width_gap)
        # sensor_fov = np.rad2deg(2 * np.arctan(width / (2 * distance)))
        sensor_fov = small_sensor.fov
        # Now, treat the small oi as a large, uniform field that covers the sensor.
        small_oi.w_angular = 2 * sensor_fov

        """
        Compute the signal voltage for this small part of the image sensor array using the brightest part of the optical 
        image, and a 1 sec exposure duration. (Generally, the voltage at 1 sec is larger than the voltage swing.)
        """
        signal_voltage = small_sensor.compute_image(small_oi)
        max_signal_voltage = np.max(signal_voltage)

        integration_time = (level * voltage_swing) / max_signal_voltage
        return integration_time

    def clear_data(self):
        """
        Clear data and noise fields stored in the sensor array.

        When parameters change and data are no longer consistent, we clear the data and various stored noise
        image fields.
        Returns:

        """
        # TOD0: Now, it seems no need to implement it.
        self.data = None
        self.offset_fpn_image = None
        self.gain_fpn_image = None
        self.col_offset_fpn_vector = None
        self.col_gain_fpn_vector = None
        self.etendue = None

    def compute_image(self, oi) -> np.ndarray:
        """
        Main routine to compute the sensor voltage data from the optical image.

        Compute the expected sensor voltage image from the sensor parameters and the optical image. It calls a variety
        of sub-routines that implement many parts of calculation. The voltages returned here are not clipped by the
        voltage swing.  That is done in then `compute` method.

        Computation steps:

        1. The current generated at each pixel by the signal is computed (signal_current). This is converted to a
        voltage using the cur2volt parameter.

        2.The dark voltage is computed and added to the signal.

        3. Shot noise is computed for the sum of signal and dark voltage signal. This noise as added.
        Hence, the poisson noise includes both the uncertainty due to the signal and the uncertainty due to the dark
        voltage. (It is possible to turn this off by setting `shot_noise_flag` to 0.)

        4. Read noise is added.

        5. If the sensor fixed pattern noises, DSNU and PRNU were previously stored, they are added/multipled into
        the signal. Otherwise, they are computed and stored and then combined in the signal.

        6. If column FPN is selected and stored, it is retrieved and combined into the signal. If columns FPN is
        selected but not stored, it is computed and applied. Finally, if it is not selected, it is not applied.

        7. Analog gain (ag) and analog offset (ao) are applied to the voltage image:
        ::
            volt_image = (volt_image + ao) / ag

        Many more notes on the calculation, including all the units are embedded in the comments below.


        Args:
            oi: The optical image

        Returns:
            volt_image: the spatial array of volts (not clipped by the voltage swing).
        """

        # q = elementary_charge
        # we use below value to reproduce the same result as the original code,
        # we should use the value from scipy.constants in the final implementation
        # Elementary charge, C

        pixel = self.pixel

        """
        Calculate current
        This factor convert pixel current to volts for this integration time. The conversion units are:
            sec * (V/e) * (e/charge) = sec * V / charge  = V / current
            
        Given the basic rule V = IR, k is the effectively a measure of the resistance that converts current into volts 
        given the exposure duration.
        We handle the case in which the integration time is a vector of matrix, by creating a matching conversion 
        from current to volts.
        """
        cur2volt = self.integration_time * pixel.conversion_gain / q
        # For now, cur2volt is a scalar. Below code is for the case when cur2volt is a vector or matrix
        if not np.isscalar(cur2volt):
            cur2volt = cur2volt.ravel()

        # Calculate the signal current assuming cur2volt = 1
        unit_sig_current = self.signal_current(oi)

        # Convert to volts
        # Handle multiple exposure vale case.
        if np.isscalar(cur2volt):
            volt_image = cur2volt * unit_sig_current

        else:
            volt_image = np.tile(unit_sig_current, (1, 1, len(cur2volt)))
            for i in range(len(cur2volt)):
                volt_image[:, :, i] = cur2volt[i] * unit_sig_current

        """
        Calculate etendue from pixel vignetting
        We want an (wavelength-independent) of scale factors that account for the loss of light at each pixel as a 
        function of the chief ray angle. This method only works for wavelength-independent relative illumination. 
        See notes in signal_current_density for another approach that we might use some day, say in ISET-3.0.
        
        """
        self.vignetting()
        etendue = self.etendue
        if self.n_exposures == 1:
            volt_image = volt_image * etendue
        else:
            volt_image = volt_image * np.tile(etendue, (1, 1, self.n_exposures))
        self.volt_image = volt_image

        # Something is wrong, return None data, including the noise images
        if self.volt_image is None:
            self.dsnu_image = None
            self.prnu_image = None
            return volt_image

        # Add the dark current
        dark_current = self.pixel.dark_current_density
        if dark_current != 0:
            """
            At this point the noise dark current is the same at all pixels. Later, we apply the PRNU gain factor the the 
            sum of the signal and noise, so that dark current effectively varies across pixels. Sam Kavusi says that 
            this variation in gain (also called PRNU) is not precisely the same for signal and noise. But we have no way 
            to access this for most cases, so we treat the PRNU for noise and signal as the same until forced to do it 
            otherwise. 
            """
            e_times = self.integration_time
            n_times = self.n_exposures
            if n_times == 1:
                volt_image = volt_image + self.pixel.dark_voltage * e_times

            else:
                for i in range(n_times):
                    volt_image[:, :, i] = volt_image[:, :, i] + self.pixel.dark_voltage * e_times[i]

        self.volts = volt_image

        """
        Add shot noise
        Note that you can turn off shot noise in the calculation by setting the shot_noise_flag to False. Default is
        True. This flag is accessed only through scripts at the moment. There is no way to turn it off from the user 
        interface.
        """
        if self.noise_flag > 0:
            volt_image = self.add_shot_noise()
            self.volts = volt_image

        # Add read noise
        noisy_image = self.add_read_noise()
        self.volts = noisy_image

        """
        noise FPN
        This combines the offset and gain (DSNU, PRNU) images with the current voltage to produce a noiser image. If 
        these images don't yet exist, we compute them.
        """
        dsnu_image = self.offset_fpn_image
        prnu_image = self.gain_fpn_image

        if dsnu_image is None or prnu_image is None:
            volt_image, dsnu_image, prnu_image = self.add_noise_FPN()
            self.dsnu_image = dsnu_image
            self.prnu_image = prnu_image
        else:
            volt_image, *_ = self.add_noise_FPN()
        self.volts = volt_image

        # Now we check for column FPN value. IF data exist then we compute column FPN noise. Otherwise, we carry on.
        if self.col_gain_fpn_vector is None and self.col_offset_fpn_vector is None:
            # Perform some action when both are None
            print("Both col_gain_fpn_vector and col_offset_fpn_vector are None")
        else:
            volt_image = self.add_column_FPN()

        """
        Analog gain simulation
        We check for an analog gain and offset. The manufactureres were clamping at zero and using the analog gain like 
        wild men, rather than exposure duration. We set it in script for now, and we will add the ability to set it in 
        the GUI before long. If these parameter are not set, we assume they are returned as 1 (gain) and 0 (offset).
         
        """
        ag = self.analog_gain
        ao = self.analog_offset
        volt_image = (volt_image + ao) / ag
        return volt_image

    def signal_current(self, oi) -> np.ndarray:
        """
        Compute the signal current at each pixel position

        The signal current is computed from the optical image and the image sensor array properties (ISA). The units
        returned are Amps/pixel = (Coulombs/sec)/pixel.

        This is a key routine called by `compute_image` and `compute`. The routine can compute the current in either
        the default spatial resolution mode (1 spatial sample per pixel) or in a high-resolution made in which the pixel
        is modeled as a grid of sub-pixels, and we integrate the spectral irradiance field across this grid weighting it
        for the light intensity and the pixel. (The latter, high resolution mode, has not been much used in years.)

        The default or high-resolution mode computation in governed by the `n_samples_per_pixel` parameter
        in the sensor.

        self.n_samples_per_pixel

        **The default mode has a value of 1 and this is the only mode we have used for many years.**
        Even so, high resolution modes can be computed with `self.n_samples_per_pixel = 5` or some other value.

        If 5 is chosen, then there is a 5x5 grid placed over the pixel to account for spatial sampling.

        Args:
            oi: The optical image

        Returns:
            signal_current: The signal current
        """

        signal_current_density_image = self.signal_current_density(oi)  # Amps/m^2

        """
        Spatially interpolate the optical image with the image sensor array. The optical image values describe the 
        incident rate of photons. It should be possible to super-sample by setting grid_spacing to, say, 0.2.
        We could do this in the user-interface some day. I am not sure that it has much benefit, but it does take a lot 
        more time and memory.
        """
        grid_spacing = 1 / self.n_samples_per_pixel
        signal_current_image = self.spatial_integration(signal_current_density_image, oi, grid_spacing)
        
        return signal_current_image


    def signal_current_density(self, oi) -> np.ndarray:
        """
        Estimate the signal current density (current/meter^2) across the sensor surface.

        This image has a spatial sampling density equal to the spatial sampling of the scene and describes the current
        per meter^2 (A/m^2).

        We perform the calculation in two ways, depending on image size. if the image is less than 512x512, we
        calculate using a quick matrix multiplication. To restrict memory use, if the optical image exceeds 512x512,
        we loop through the wavelengths. Slower but it means the memory used stays below 64MB.
        Computation steps:

        1. The irradiance image in photons is multiplied by the spectral QE information.

        2. The calculation treats the input data as photons, estimates the fraction of these that are effective, and
        then turns this into a charge per unit area.

        3. Subsequent calculations account for the photodetector area.

        There are many comments in the code explaining each step.

        Args:
            oi: optical image

        Returns:
            scd_image: signal current density image
        """
        # optical image variables
        bin_width: float = oi.bin_width
        n_rows = oi.rows
        n_cols = oi.cols
        oi_wave = oi.wave
        n_wave = len(oi_wave)

        # sensor variables
        n_filter = self.n_filter
        spectral_qe: np.ndarray = self.color['filter_spectra']  # [wavelength, n_filters]
        sensor_wave = self.wave

        """
        It is possible that the sensor spectral QE is not specified at the same wavelength sampling resolution as the 
        irradiance. In that case, we resample to the lower wavelength sampling resolution.
        """
        if n_wave != len(sensor_wave):
            f = interp1d(sensor_wave, spectral_qe, kind='linear', bounds_error=False, fill_value=0)
            spectral_qe = f(oi_wave)

        """
        At this point, the spectral quantum efficiency is defined over wavelength bins of size, bin_width.
        To count the number photons in the entire bin, we must multiply by the bin width.
        """
        s_qe = spectral_qe * bin_width  # [wavelength, n_filters] * scalar = [wavelength, n_filters]

        """
        Sensor etendue: In all ISET calculations we treat the etendue (i.e. the pixel vignetting) as if it is wavelength 
        independent. This is an OK approximation.
        But if we ever want to treat etendue as a function of wavelength, we sill have to account for it at this point, 
        before we collapse all the wavelength information into a single number (the signal current density).
        
        If we do that, we may need a space-varying calculation. That would be computationally expensive. We aren't yet 
        ready for that level of detail.
        
        At present the etendue calculation is incorporated as a single scale factor at each pixel and incorporated in 
        the `self.compute` routine.
        
        s_qe is a wavelength x n_filters matrix, and it includes a conversion factor that will maps the electrons that 
        will maps the electrons per square meter into amps per square meter.
        
        Multiply the optical image with the photodetector QE and the color filters. Accumulated this way, we form a 
        current density image at every position for all the color filters.
        output units: A/m^2
        
        Critical size used to decide which computational method is applied. The computational issue is memory size 
        versus speed (see below).
        Changed 4x in 2015 because computers are bigger.
        Changed again in 2022 because computers are bigger (again:)).
        """
        crit_size = 2 ** 26

        if n_rows * n_cols < crit_size:
            """
            This is faster. But if we are trying to limit the memory size, we should use the other part of the loop 
            that calculates one waveband at a time.
            """
            irradiance = oi.photons  # quanta/m2/nm/sec, left single(?)
            irradiance, *_ = rgb_to_xw(irradiance)
            scd_image = irradiance @ s_qe  # sum_bin (quanta/m2/nm/sec * nm/bin) = (quanta/m2/sec)
            scd_image = xw_to_rgb(scd_image, n_rows, n_cols)
            """
            At this point, if we multiply by the photodetector area and the integration time, that gives us the number 
            of electrons at a pixel.
            """
        else:
            # For large images, don't take all the data out at once. Do it a waveband at a time.
            scd_image = np.zeros((n_rows, n_cols, n_filter))
            for i in range(n_wave):
                irradiance = oi.photons[:, :, i]
                irradiance = rgb_to_xw(irradiance)
                scd_image[:, :, i] = irradiance @ s_qe[i, :]

        """
        Convert the photons into a charge using the constant that defines charges/electron. This is the signal current 
        density (scd) image. It has units of quanta/m2/sec/bin * charge/quanta = charge/m2/sec/bin.
        """
        scd_image = scd_image * q
        return scd_image

    def spatial_integration(self, scdi: np.ndarray, oi, grid_spacing: float = 0.2) -> np.ndarray:
        """
        Measure current at each sensor photodetector

        The signal current density image(scdi) specifies the current (A/m^2) across the sensor surface at a set of
        sample values specified by the optical image. This routine converts that scdi to a set of currents at each
        photodetector.

        The routine can operate in two modes. In the first (lower resolution, fast, default) mode, the routine assumes
        that each photodetector is centered in the pixel. In the second (higher resolution, slow) mode, the routine
        accounts for the position and size of the photodetectors within each pixel.

        The sensor pixels define a coordinate frame that can be measured (in units of meters). The optical image also
        has a size that can be measured in meters. In both modes, we represent the OI and the ISA sample positions in
        meters in a spatial coordinate frame with a common center. Then we interpolate the values of the OI onto sample
        points within the ISA grid (regrid_oi2isa).

        The first mode (default). In this mode the current is computed with one sample per pixel. Specifically, the
        irradiance at each wavelength is linearly interpolated to obtain a value at the center of the pixel.

        The second mode (high-resolution). This high-resolution mode requires a great deal more memory than the first
        mode. In this method a grid is placed over the sensor and the irradiance field in interpolated to every point in
        that grid(e.g., 5x5 grid). The pixel is computed by summing across those grid points (weighted appropriately).

        The high-resolution mode used to be the default mode. But over time we came to believe that it is better to
        understand the effects of photodetector placement and pixel optics using the microlenswindow module. For
        certain applications, though, such as illustrating the effects of wavelength-dependent point spread functions,
        this mode is valuable.

        Args:
            scdi: signal current density image
            oi: optical image
            grid_spacing: grid spacing

        Returns:
            signal_current_image: the current at each photodetector
        """
        """
        We can optionally represent the scdi and imager at a finer resolution than just pixel positions. 
        This permits us to account for the size and position of the photodetector within the pixel. 
        To do this, however, requires that we regrid the signal current density image to a finer scale. 
        To do this, the parameter 'spacing' can be set to a value of, say, .2 = 1/5. 
        In that case, the super-sampled new grid is 5x in each dimension.  
        This puts a large demand on memory usage, so we don't normally do not.  
        Instead, we use a default of 1 (no gridding).
        
        This is the spacing within a pixel on the sensor array.
        """
        grid_spacing = 1 / np.round(1 / grid_spacing)

        n_grid_samples = 1 / grid_spacing

        """
        regried_oi2isa puts the optical image pixel in the same coordinate frame as the sensor pixels. The sensor pixel 
        coordinate frame is simply the pixel position (row, col). If grid_spacing = 1 , then there is a one-to-one 
        matching between the pixels and calculated signal current density image. if grid_spacing is smaller, say 0.2, 
        then there are more grid samples per pixel. this can pay a significant penalty in speed and memory usage.
        
        So the default grid spacing is 1.
        """
        flat_scdi = self.regrid_oi2isa(scdi, oi, grid_spacing)

        """
        Calculate the fractional area of the photodetector within each grid region of each pixel. 
        If we are super-sampling, we use sensor_pd_array. Otherwise, we only need the fill factor.
        """
        if n_grid_samples == 1:
            pd_array = self.pixel.fill_factor
        else:
            pd_array = self.sensor_pd_array(grid_spacing)
        isa_size = self.size
        photo_detector_array = np.tile(pd_array, isa_size)
        signal_current_image_large = flat_scdi * photo_detector_array

        if n_grid_samples == 1:
            signal_current_image = self.pixel.area * signal_current_image_large

        else:
            # if the grid samples are super-sampled, the image must be collapsed by summing across the pixel and create
            # an image that has the same size as the ISA array. This is done by blur_samples routine. 
            
            
            filt = self.pixel.area * np.ones((n_grid_samples, n_grid_samples)) / n_grid_samples ** 2
            signal_current_image = self.blur_samples(signal_current_image_large, filt, 'same')

        return signal_current_image

    def sensor_pd_array(self, grid_spacing: float) -> np.ndarray:
        raise NotImplementedError('This function is not implemented yet')

    def regrid_oi2isa(self, scdi, oi, grid_spacing):
        """
        Regrid current density in OI coordinates into sensor coordinates.

        The spatial samples in the signal current density image are represented on a grid determined by the optical
        image sampling. This routine regrids the scdi spatial samples into the spatial sample positions of the sensor
        pixels.

        Both the scdi (and OI) and the sensor samples are specified in units of microns, this routine linearly
        interpolates the scdi samples to a grid where integer values correspond to the spatial sample positions of the
        sensor pixel. The routine uses matlab's interp1 routine.


        Args:
            scdi: signal current density image
            oi: optical image
            grid_spacing:

        Returns:
            flat_scdi: the resampled signal current density image at the spatial sampling resolution of the sensor.
        """

        r, c = oi.size
        r_samples = np.arange(0, r)
        c_samples = np.arange(0, c)
        oi_height_spacing = oi.get_h_spatial_resolution()
        oi_width_spacing = oi.get_w_spatial_resolution()

        # Puts the number of rows and columns in units of microns
        these_rows, these_cols = sample_to_space(r_samples, c_samples, oi_height_spacing, oi_width_spacing)
        U, V = np.meshgrid(these_cols, these_rows)

        """
        The values of new_rols and new_cols are sampled positions on the image sensor array. If spacing < 1. they are 
        spaced more finely than the pixel samples. We haven't done a lot of calculations in recent years with spacing 
        < 1. For some cases, this could be an issue - maybe for a very small point in the oi.
        """
        r, c = self.size
        r_samples = np.arange(0, r, grid_spacing) + grid_spacing / 2
        c_samples = np.arange(0, c, grid_spacing) + grid_spacing / 2

        sensor_height_spacing = self.deltay
        sensor_width_spacing = self.deltax

        new_rows, new_cols = sample_to_space(r_samples, c_samples, sensor_height_spacing, sensor_width_spacing)
        X, Y = np.meshgrid(new_cols, new_rows)

        # Initialize the signal current density image on the planer surface of the sensor.
        flat_scdi = np.zeros_like(X)
        n_filter = self.n_filter

        # Determine the color filter number at each point on the sensor surface.
        interpolated_cfa_n = self.interp_cfa_scdi(new_rows, new_cols, grid_spacing)

        # We add a Gaussian blurring kernel. We

        ### FIXME: sigma should be scalar but we got 2d array so .item() is used but should be fixed
        height_samples_per_pixel = np.ceil(sensor_height_spacing / oi_height_spacing).item()
        width_samples_per_pixel = np.ceil(sensor_width_spacing / oi_width_spacing)

        kernel_size = [int(height_samples_per_pixel), int(width_samples_per_pixel)]
        kernel = np.zeros(kernel_size)

        # Set the middle element to 1
        kernel[kernel_size[0] // 2, kernel_size[1] // 2] = 1

        # Apply the Gaussian filter
        g_kernel = gaussian_filter(kernel, sigma=height_samples_per_pixel / 4)
        if r > 100:
            tt = 0
        for i in range(n_filter):
            scdi_i = scdi[:, :, i]
            scdi_i = convolve2d(scdi_i, g_kernel, mode='same')

            u = U[0, :]
            v = V[:, 0]
            f = interp2d(u, v, scdi_i, kind='linear', bounds_error=False, fill_value=0)
            x = X[0, :]
            y = Y[:, 0]
            tmp = f(x, y)  # U, V는 원본 matlab code 대비 transposed 되어 있다.

            # from scipy.io import loadmat
            # tmp_mat = loadmat('check_data/tmp.mat')['tmp']
            # assert np.allclose(tmp, tmp_mat.T), 'The result is not same as the expected one.'

            mask = interpolated_cfa_n == (i + 1)

            if (tmp.shape[0] == 1) or (tmp.shape[1] == 1):
                tmp = tmp.reshape(mask.shape)

            flat_scdi = flat_scdi + mask * tmp
        flat_scdi = np.nan_to_num(flat_scdi)
        return flat_scdi

    def interp_cfa_scdi(self, r_pos, c_pos, spacing):
        """
        This routine determines the color filter at each r_pos, c_pos values in the coordinate frame of the sensor.
        The positions do not need to be at the grid positions of the pixels, but this routine makes a best estimate of
        the color at each position. The algorithm in here could be adjusted based on various factors, such as microlens
        in the future. At present, we are just rounding. The integer values of the CFA are determined by the sensor
        color filter names and the definition in sensor_color_order.
        Args:
            r_pos: row position
            c_pos: column position
            spacing: grid spacing

        Returns:
            interpolated_cfa_n: interpolated cfa number

        """

        cfa, cfa_n = self.determine_cfa()
        r_coords = np.floor(np.arange(len(r_pos)) * spacing)
        c_coords = np.floor(np.arange(len(c_pos)) * spacing)
        interpolated_cfa_n = cfa_n[np.ix_(r_coords.astype(int), c_coords.astype(int))]
        # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

        return interpolated_cfa_n

    def determine_cfa(self):
        """
        Determine the CFA organization for an image sensor.

        The CFA letters in an array of sensor size containing a string that characterizes the filter appearance. If the
        filter_name first letter that is written the list defined in `self.color_order`, then we use that letter.
        Otherwise, we assign the color k (black). This may encourage better notation. Or, we may get annoyed with this
        and write a routine to determine a reasonable color from the filter_spectra curve.

        The cfa_number is an integer matrix with the size of sensor. The integers in the matrix refer to which color
        filter (as defined by the colum numbers occupied by the color filter in the sensor data structure
        (self.filter_spectra)). These are the same numbers that are used by the field 'pattern' to define the color
        filter array spatial layout.

        Returns:

        """
        rows = self.rows
        cols = self.cols

        pattern = self.cfa_pattern
        if pattern.shape[0] != rows or pattern.shape[1] != cols:
            block_rows = self.unit_block_rows
            block_cols = self.unit_block_cols

            r_factor = rows // block_rows
            c_factor = cols // block_cols
            cfa_numbers = np.tile(pattern, (r_factor, c_factor))
            cfa_numbers = cfa_numbers[:rows, :cols]

        else:
            r_factor = 1
            c_factor = 1
            cfa_numbers = pattern

        """
        Create the list of characters that are hints to the color appearance we should assign to each color filter. 
        These hints can be useful for plotting routines. Get the letters from the first character of the filter name.
        """
        filter_color_letters = self.filter_color_letters

        pattern_colors = self.pattern_colors

        cfa_letters = np.tile(pattern_colors, (r_factor, c_factor))
        return cfa_letters, cfa_numbers

    def vignetting(self, n_angles: float | None = None, pv_flag: int = 0):
        """
        Gateway routine for computing pixel vignetting.

        This is a gateway routine to ml_analyze_array_etendue. The pixel vignetting information from the micro-lens is
        attached to the image sensor array.

        Vignetting and in this case etendue refers to the loss of light sensitivity that depends on the location of the
        pixel with respect to the principal axis of the imaging lens. The effects of vignetting are calculated in the
        microlens (ml) functions.

        Args:
            n_angles: Number of angles for the etendue calculation
            pv_flag: Pixel vignetting flag

        Returns:

        """

        match pv_flag:
            case 0:
                sz = self.size
                self.etendue = np.ones(sz)
            case 1:  # bare, no micro-lens
                self.ml_analyze_array_etendue('no microlens', n_angles)
            case 2:  # centered
                self.ml_analyze_array_etendue('centered', n_angles)
            case 3:  # optimal
                self.ml_analyze_array_etendue('optimal', n_angles)
            case _:
                raise ValueError(f'Unknown pixel vignetting flag {pv_flag}...')

    def add_shot_noise(self) -> None:
        """
        Add shot noise (Poisson electron noise) into the image data.

        The shot noise is Poisson in the units of electrons (but not in other units). Hence, we transform the (mean) voltage
        image to electrons, create the Poisson noise, and then the signal back to a voltage. The returned noisy voltage
        signal is not Poisson; it has the same SNR (mean/sd) as the electron image.

        This routine uses the normal approximation to the Poisson when there are more than 25 electrons in the pixel.

        Returns:
            None

        """
        # Get the electrons from the sensor
        electrons = self.electrons

        electrons_noise = np.sqrt(electrons) * np.random.randn(*electrons.shape)
        poisson_criterion = 25

        # TODO: Implement, when the number of electrons is less than the poisson_criterion

        conversion_gain = self.pixel.conversion_gain

        noisy_image = conversion_gain * np.round(electrons + electrons_noise)
        the_noise = conversion_gain * electrons_noise
        return noisy_image

    def add_read_noise(self) -> np.ndarray:
        """
        Add read noise to the sensor image.

        The read noise is a Gaussian random variable.

        Returns:
            noisy_image: The noisy image

        """
        read_noise = self.pixel.read_noise_volts
        noisy_image = self.volts + np.random.randn(*self.size) * read_noise
        return noisy_image

    def add_noise_FPN(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Include dsnu and prnu noise into a sensor image.

        This routine adds the dsnu and prnu noise.
        The DSNU and PRNU act as an additive offset to the voltage image (DSNU) and as a multiplicative gain factor
        (PRNU). Specifically, we first compute the mean voltage image. Then we transform the mean using compute the
        mean voltage image. Then we transform the mean using
        ::
            output_vooltage = (1 + prnu) * mean_voltage + dsnu

        where DSNU is a Gaussian random variable with a standard deviation obtained by self.dsnu_sigma. The PRNU is also
        a Gaussian random variable with a standard deviation of self.prnu_sigma. The dsnu_sigma and prnu_sigma are set
        in the sensor window interface.

        This routine permits a zero integration time so that it can be used for CDS calculations. In this case, when
        self.integration_time = 0, no prnu_image is returned because, well, there is no gain.

        Returns:
            noise_image: The noisy image

            dsnu_image: The dark signal non-uniformity image

            prnu_image: The photo response non-uniformity image

        """

        sz = self.size
        gain_sd = self.sigma_gain_fpn
        offset_sd = self.sigma_offset_fpn
        integration_time = self.integration_time
        ae = self.auto_exposure

        # Get the fixed pattern noise offset
        dsnu_image = np.random.randn(*sz) * offset_sd

        """
        For CDS calculations, we can arrive here with all the integration times are 0 and auto_exposure is off.
        We do a special calculation.
        """
        if integration_time == 0 and not ae:
            noise_image = dsnu_image
            prnu_image = np.random.randn(*sz) * (gain_sd / 100 + 1)


        else:
            # TODO: Add comments for the below code
            prnu_image = np.random.randn(*sz) * (gain_sd / 100) + 1

            n_exposures = self.n_exposures

            voltage_image = self.volts
            noise_image = voltage_image * prnu_image + dsnu_image

        return noise_image, dsnu_image, prnu_image

    def add_column_FPN(self) -> np.ndarray:
        """
        Apply column fpn to the voltage in the sensor image.

        The column offset (DSNU) is a Guassian random variable. The column gain (PRNU) is a random variably
        around unit slope, i.e.,
        ::
            col_prnu = N(0, 1) * col_prnu + 1

        If the column FPN values are not already computed, they are computed here. Usually, however, the values are
        computed in sensor_window.
        Returns:
            noise_image: The noisy image

        """
        n_col = self.cols
        n_row = self.rows

        col_offset_fpn = self.col_offset_fpn_vector
        col_gain_fpn = self.col_gain_fpn_vector
        voltage_image = self.volts

        if col_offset_fpn != 0 or col_gain_fpn != 0:
            col_dsnu = np.random.randn(1, n_col) * col_offset_fpn
            col_prnu = np.random.randn(1, n_col) * col_gain_fpn + 1
            noise_image = voltage_image @ np.diag(col_prnu) + np.tile(col_dsnu, (n_row, 1))

        else:
            noise_image = voltage_image
        return noise_image

    def add_noise(self):
        """
        Add electrical and photon noise to then sensor voltage image.
        Typical
        Returns:

        """
        pixel = self.pixel
        noise_flag = self.noise_flag
        if noise_flag == 0:
            return
        self.reuse_noise = False
        if self.reuse_noise:
            raise NotImplementedError('This function is not implemented yet')
        else:
            np.random.seed(42)

        # perform the noise addition steps here
        n_exposures = self.n_exposures
        e_times = self.integration_time
        volts = self.volts

        for i in range(n_exposures):
            # v_image = volts[:, :, i]
            v_image = volts

            if noise_flag > 1:
                v_image = v_image + pixel.dark_voltage * e_times
                self.volts = v_image

            if noise_flag > 0:
                self.add_shot_noise()

            if noise_flag > 1:
                v_image = v_image + pixel.read_noise_volts * np.random.randn(*v_image.shape)
                self.volts = v_image

                v_image, *_ = self.add_noise_FPN()
                self.volts = v_image

                self.col_offset_fpn_vector, self.col_gain_fpn_vector = self.column_fpn
                v_image = self.add_column_FPN()
                self.volts = v_image

        self.volts = v_image

    def get_image(self, gamma: float = 1.0, scale_max: float = 0.0) -> np.ndarray:
        """
        Display the image in a scene.
        # TODO: Write comments

        Args:
            gamma: display gamma
            scale_max:  scale to maximum brightness
        Returns:
            image: rendered image
        """

        img = self.data2image('dv or volts', gamma, scale_max)

        # TODO: Implement the below part
        p_size = self.pixel.size
        row, col = self.size
        x = np.arange(0, col)
        y = np.arange(0, row)
        s_factor = p_size[0] / p_size[1]
        if s_factor > 1:
            y = y * s_factor
        else:
            x = x / s_factor

        return img

    def data2image(self, data_type: str, gamma: float, scale_max: float) -> np.ndarray:
        """
        Produce the image data displayed in the window.

        This routine creates the color at each pixel resemble the transmissivity of the color filter at that pixel.
        The intensity the size of the data. The data_type is normally volts.

        Normally, the functions takes in one CFA plane. It can also handle the case of multiple exposure durations.

        While it is usually used for volts, the routine converts the image from the 'dv' fields or even 'electrons'.

        The returned images can be written out as a tiff file by save_image.

        Args:
            data_type: data type, default is 'dv or volts'
            gamma: display gamma
            scale_max: scale to maximum brightness

        Returns:
            image: rendered image
        """
        data_type = data_type.replace(' ', '_')
        img = getattr(self, data_type)
        if scale_max:
            max_image = np.max(img)
        else:
            match data_type:
                case 'volts':
                    max_img = self.max_output
                case 'dv_or_volts':
                    try:
                        max_img = self.max_digital_values
                    except AttributeError:
                        max_img = self.max_output
                case _:
                    raise ValueError(f'Unknown data type {data_type}...')

        n_sensor = self.n_filter
        exp_times = self.integration_time
        n_exposures = self.n_exposures

        if n_exposures > 1:
            p_size = self.pattern_colors
            # TODO: Implement this part
            raise NotImplementedError('This function is not implemented yet')

        if n_sensor > 1:  # A color CFA

            img = self.plane2rgb(img, empty_value=0)

            match self.filter_color_letters:
                case 'rgb':
                    pass
                case 'wrgb':
                    raise NotImplementedError('This function is not implemented yet')
                case 'rgbw':
                    raise NotImplementedError('This function is not implemented yet')
                case _:
                    raise ValueError(f'Unknown filter color letters {self.filter_color_letters}...')

            img = img / max_img
            img = np.clip(img, 0, 1) ** gamma

        img = np.clip(img, 0, 1)
        if img.shape[-1] == 3:
            img = lrgb2srgb(img)
        return img

    def plane2rgb(self, img: np.ndarray, empty_value: np.NAN = np.NAN) -> np.ndarray:
        """
        Convert a sensor data plane (r,c) in to RGB image (r, c, 3)

        # TODO: Write docstring
        Args:
            img: The vc_image input field, which typically the data field from the sensor
            empty_value: default is NaN. In the RGB format the unfilled values are all set to empty_value

        Returns:

        """
        cfa, cfa_n = self.determine_cfa()
        filter_color_letters = self.filter_color_letters
        n_planes = len(filter_color_letters)
        rows, cols = img.shape
        rgb_format = np.zeros((*img.shape, 3))

        assert img.shape == cfa.shape, 'The shape of the input image and the CFA pattern should be same.'

        for i in range(n_planes):
            tmp = np.ones((rows, cols)) * empty_value
            l = cfa_n == (i + 1)
            tmp[l] = img[l]
            rgb_format[:, :, i] = tmp.reshape(rows, cols)

        return rgb_format

    def color_order(self, format='cell'):
        cfa_ordering = ('r', 'g', 'b',
                        'c', 'y', 'm',
                        'w', 'i', 'u',
                        'x', 'z', 'o', 'k')
        # The ordering of these color map entries must map cfa_ordering, above.
        cfa_map = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (0, 1, 1), (1, 1, 0), (1, 0, 1),
                   (1, 1, 1), (.3, .3, .3), (.4, .7, .3),
                   (.9, .6, .3), (.2, .5, .8), (1, .6, 0), (0, 0, 0))
        if format == 'string':
            cfa_ordering = ''.join(cfa_ordering)
        return cfa_ordering, cfa_map

    def ml_analyze_array_etendue(self, ml_type: str, n_angles: int | None):
        """
        Analyze the etendue across a sensor array.

        # TODO: Implement this method

        Args:
            ml_type:
            n_angles:

        Returns:

        """
        raise NotImplementedError('This function is not implemented yet')

    @staticmethod
    def blur_samples(signal_current_image_large: np.ndarray, filt: np.ndarray, mode: str) -> np.ndarray:
        raise NotImplementedError('This function is not implemented yet')


    @property
    def cfa_size(self):
        return self.cfa_pattern.shape

    @property
    def fov(self) -> float:
        try:
            scene = self.scene
            s_distance = scene.s_distance
        except AttributeError:
            s_distance = 1e6

        distance = self.oi.get_optics_focal_plane_distance(s_distance)
        width = self.width
        fov = np.rad2deg(2 * np.arctan(width / (2 * distance)))
        return fov

    @property
    def width(self) -> float:
        return self.cols * self.deltax
    
    @property
    def n_filter(self):
        """Get the number of color filters in the sensor array.
        normally filter_spectra array shape is [wavelength, n_filters], So we get the last dimension"""
        return self.color['filter_spectra'].shape[-1]

    @property
    def deltax(self) -> float:
        pixel = self.pixel
        val = pixel.width + pixel.width_gap
        return val

    @property
    def deltay(self) -> float:
        pixel = self.pixel
        val = pixel.height + pixel.height_gap
        return val

    @property
    def unit_block_rows(self) -> int:
        return self.cfa_pattern.shape[0]

    @property
    def unit_block_cols(self) -> int:
        return self.cfa_pattern.shape[1]

    @property
    def filter_color_letters(self) -> str:
        names = self.color['filter_names']
        names = [name[0] for name in names]
        names = ''.join(names)
        return names

    @property
    def pattern_colors(self):
        pattern = self.cfa_pattern
        filter_color_letters = self.filter_color_letters
        filter_color_letters = list(filter_color_letters)  # Convert to list for assignment
        known_color_letters, _ = self.color_order(format='string')
        known_filters = [letter in known_color_letters for letter in filter_color_letters]
        # Assign unknown color filter strings to black ('k')
        for i, is_known in enumerate(known_filters):
            if not is_known:
                filter_color_letters[i] = 'k'
        # filter_color_letters = ''.join(filter_color_letters)
        # Create a block that has letters instead of numbers
        val = np.array([[filter_color_letters[i - 1] for i in row] for row in pattern])
        return val

    @property
    def n_exposures(self):
        if isinstance(self.integration_time, (int, float)):
            return 1
        else:
            return len(self.integration_time)

    @property
    def electrons(self):
        pixel = self.pixel

        ag = self.analog_gain
        ao = self.analog_offset
        cg = pixel.conversion_gain
        electrons = (self.volts * ag - ao) / cg
        return np.round(electrons)

    @property
    def column_fpn(self) -> tuple[float, float]:
        try:
            return self.col_offset_fpn_vector, self.col_gain_fpn_vector
        except AttributeError:
            return 0, 0

    @property
    def dv_or_volts(self) -> np.ndarray:
        try:
            return self.dv
        except AttributeError:
            return self.volts

    @property
    def max_output(self) -> float:
        pixel = self.pixel
        return pixel.voltage_swing


if __name__ == '__main__':
    illuminant = Illuminant(illuminant_name = 'd65')
    scene = Scene(scene_name = 'macbeth', illuminant = illuminant, hfov = 5)
    oi = OI('fisheye', scene)
    oi.optics_ray_trace()
    
    pixel = Pixel(pixel_type='default')
    sensor = Sensor(sensor_type='bayer', pixel = pixel)

    # Sensor parameters
    voltage_swing = 1.15  # Volts
    well_capacity = 9000  # Electrons
    
    pixel_size = 2.2 * 1e-6  # Meters
    conversion_gain = voltage_swing / well_capacity
    fill_factor = 0.45  # A fraction of the pixel area
    dark_voltage = 1e-005  # Volts/sec
    read_noise = 0.00096  # Volts

    # We set these properties here
    pixel.size = (pixel_size, pixel_size)
    pixel.conversion_gain = conversion_gain
    pixel.voltage_swing = voltage_swing
    pixel.dark_voltage = dark_voltage
    pixel.read_noise_volts = read_noise

    sensor.pixel = pixel

    # Now we set some general sensor properties
    # exposure_duration = 0.030  # commented because we set autoexposure
    rows = 466  # number of pixels in a row
    cols = 642  # number of pixels in a column
    dsnu = 0.0010  # Volts (dark signal non-uniformity)
    prnu = 0.2218  # Percent (ranging between 0 and 100) photodetector response non-uniformity
    analog_gain = 1  # Used to adjust ISO speed
    analog_offset = 0  # Used to account for sensor black level

    # sensor.auto_exposure = 1
    sensor.rows = rows
    sensor.cols = cols
    sensor.sigma_offset_fpn = dsnu
    sensor.sigma_gain_fpn = prnu
    sensor.analog_gain = analog_gain
    sensor.analog_offset = analog_offset
    
    sensor.compute(oi)
    sensor_img = sensor.get_image(gamma = 1)
    
    import matplotlib.pyplot as plt
    plt.imshow(sensor_img)
    plt.show()
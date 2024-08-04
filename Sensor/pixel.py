# Pip-installed packages
import numpy as np
from typing import Literal, Sequence



place_type = Literal['center', 'corner']
ELECTRON_CHARGE = 1.602177e-19  # Coulombs

class Pixel:
    """
    The pixel class describes the pixel parameters.

    We initialize the values for simplicity and the user sets values from data within their own environment.
    For example, the photodetector is initialized to a spectral QE of 1.0 at all wavelengths.

    At present, we create these default pixel types:
        'aps', 'default', a 2.8 um active pixel sensor
        'human cone', a model of the human cone mosaic
        'mouse cone'
        'ideal', 100% fill factor, 1.5 micron, see below for the rest.
    """
    
    PIXEL_TYPES = Literal['ideal', 'default', 'aps', 'human', 'human cone', 'mouse', 'mouse cone']

    def __init__(self,
                 pixel_type: PIXEL_TYPES = 'default',
                 wave: np.ndarray = np.arange(400, 701, 10),
                 pixel_size_m: float = 2.8e-6,
                 ):
        """
        Create a pixel instance.
        Args:
            pixel_type: The type of pixel to create, default is 'default'
            wave: The wavelength samples, default is 400:700:10
            pixel_size_m: size of the pixel in meters, default is 2.8 um (2.8e-6)
        """
        self.pixel_type = pixel_type
        self.wave = wave
        self.pixel_size_m = pixel_size_m

        # Initialize the pixel parameters in pixel_aps_init
        self.name: str | None = None
        self.width: float | None = None
        self.height: float | None = None
        self.width_gap: float | None = None
        self.height_gap: float | None = None
        self.default_fill_factor: float | None = None
        self.pd_width: float | None = None
        self.pd_height: float | None = None

        self.conversion_gain: float | None = None

        self.voltage_swing: float | None = None
        self.dark_voltage: float | None = None
        self.read_noise_volts: float | None = None
        self.refractive_indices: np.ndarray | None = None
        self.layer_thickness: np.ndarray | None = None

        # Initialize parameters in get_pixel_position_pd
        self.pd_x: float | None = None
        self.pd_y: float | None = None

        match self.pixel_type:
            case 'ideal':
                pass
            case 'default' | 'aps':
                self.pixel_aps_init()
            case 'human' | 'human cone':
                self.pixel_human()
            case 'mouse' | 'mouse cone':
                self.pixel_mouse()
            case _:
                raise ValueError(f'Pixel type {self.pixel_type} not recognized')

        self.pd_spectral_qe = np.ones_like(self.wave)  # initialize to 1.0 at all wavelengths

    def pixel_aps_init(self) -> None:
        """
        A typical 2.8 um active pixel sensor.
        Returns:
            None
        """

        self.name = 'aps'
        self.width = 2.8e-6
        self.height = 2.8e-6
        self.width_gap = 0
        self.height_gap = 0

        self.default_fill_factor = 0.75
        pd_size = np.sqrt(self.width * self.height * self.default_fill_factor)
        self.pd_width = pd_size
        self.pd_height = pd_size

        self.get_pixel_position_pd()

        self.conversion_gain = 1e-4  # [V/e-]
        self.voltage_swing = 1  # [V]

        """
        Dark current density defines how quickly the pixel fill up with dark current.  The units are [A/m^2]
        In electrons per pixel per second: dk_cur_dens * pd_area / q : (chg / sec) / m^2 * m^2 / (chg / e-)
        In volts per pixel per sec: (dk_cur_dens * pd_area / q) * conversion_gain : 
        (chg / sec) / m^2 * m^2 / (chg / e-) * (V / e-)
        
        We set the density so that the well-capacity fills up with from dark current in 10 sec.
        This means in volts/pix/sec we want
            0.1 = (dk_cur_dens * pd_area / q) * conversion_gain
        so: dk_cur_dens = 0.1  * q / (pd_area * conversion_gain), units are [A/m^2]
        
        We want the dark voltage to fill up the voltage swing in 10 sec.
        So, desired is: dark_voltage_per_sec = voltage_swing / 10
        In terms of current density, the dark voltage per pixel per sec is: 
        dark_voltage = conversion_gain * dk_cur_dens * pd_area / q
        (voltage_swing / 10) (q / (con_gain * pd_area)) = dk_cur_density
        """

        # V / sec * chg / e- / (m^2 * V / e-) = (chg / sec) / m^2 = A / m^2
        self.dark_voltage = self.voltage_swing / 1000

        # 1 millivolt against 1 V total swing. Not much
        self.read_noise_volts = 0.001

        # Always starts with air and ends with silicon. We assume in between is silicon nitride and oxide.
        self.refractive_indices = np.array([1, 2, 1.46, 3.5])

        # These thicknesses make the pixel 7 microns high. we think they are around 9, but Microns is shorter.
        self.layer_thickness = np.array([2, 5]) * 1e-6  # In microns.  Air and material are infinite.

    def pixel_human(self) -> None:
        """
        A typical human cone properties
        Returns:
            None
        """

        self.name = 'humancone'
        
        # Human cones are 2 um, roughly, just outside the foveola
        self.width = 2e-6
        self.height = 2e-6
        self.width_gap = 0
        self.height_gap = 0

        # Make 100 percent fill factor
        self.pd_width = 2e-6
        self.pd_height = 2e-6
        self.get_pixel_position_pd()

        # Specifies the dynamic range effectively
        self.conversion_gain = 1e-5  # [V/e-]
        self.voltage_swing = 1  # [V]

        # Noise properties
        self.dark_voltage = self.voltage_swing / 1000

        # 1 millivolt against 1 V total swing. Not much
        self.read_noise_volts = 0.001

        # Always starts with air and ends with silicon. We assume in between is silicon nitride and oxide.
        self.refractive_indices = np.array([1, 2, 1.46, 3.5])

        # These thicknesses make the pixel 5 microns high
        self.layer_thickness = np.array([0.5, 4.5]) * 1e-6  # In microns.
        
    def pixel_mouse(self) -> None:
        """
        A typical mouse cone
        
        The mouse has no fovea, this is any retina cone. The cones are much sparser than on the human fovea, 
        since there are many rods interspaced with cones
        
        Data source: "The Major Cell Populations of the Mouse Retina", Jeon Strettoi Masland, 1998
        
        Returns:
            None
        """

        self.name = 'mousecone'
        
        self.width = 9e-6
        self.height = 9e-6
        self.width_gap = 0
        self.height_gap = 0

        # Photodetector size is same as the cone size
        self.pd_width = 2e-6
        self.pd_height = 2e-6

        self.get_pixel_position_pd()

        # Specifies the dynamic range effectively
        self.conversion_gain = 1e-5  # [V/e-]
        self.voltage_swing = 0.2  # [V]

        # Noise properties
        self.dark_voltage = 0   # No dark noise
        
        # 1 millivolt against 1 V voltage swing
        self.read_noise_volts = 0   # No read noise

    def get_pixel_position_pd(self, place: place_type = 'center') -> None:
        """
        Place the photodetector with a given height and width at the center of the pixel.
        This routine places the photodetector upper left cornet positions.
        
        Args:
            place: 'center' or 'corner'

        Returns:
            None
        """

        match place:
            case 'center':
                self.pd_x = (self.width - self.pd_width) / 2
                self.pd_y = (self.height - self.pd_height) / 2
                if self.pd_x < 0 or self.pd_y < 0:
                    raise ValueError('Inconsistent photodetector and pixel sizes.')
            case 'corner':
                self.pd_x = 0
                self.pd_y = 0
            case _:
                raise ValueError(f'Place {place} not recognized')

    @property
    def pd_size(self) -> tuple[float, float]:
        return self.pd_height, self.pd_width

    @property
    def pd_area(self) -> float:
        return self.pd_height * self.pd_width

    @property
    def fill_factor(self) -> float:
        return self.pd_area / self.area

    @property
    def spectrum(self) -> np.ndarray:
        return self.wave

    @spectrum.setter
    def spectrum(self, wave: np.ndarray) -> None:
        self.wave = wave

    @property
    def size(self) -> tuple[float, float]:
        return self.width, self.height

    @size.setter
    def size(self, size: Sequence[float]) -> None:
        assert len(size) == 2, 'Size must be a tuple of two values'
        self.width, self.height = size

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def dark_current_density(self) -> float:
        return self.dark_current / self.pd_area

    @property
    def dark_current(self) -> float:
        return self.dark_voltage / self.conversion_gain * ELECTRON_CHARGE
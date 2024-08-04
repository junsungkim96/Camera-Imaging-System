import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pip-installed packages
import numpy as np

# Local modules
from Utils.utils import energy_to_quanta, read_spectral, luminance_from_energy


class Illuminant():
	"""
	Create an illuminant(light source) structure
	
	Args:
		illuminant_file (str): Repository in which the illuminant data are stored
		wave (np.ndarray): List of wavelengths
		colorTemp: Required when the illuminant is a blackbody (deg K)
		luminance: Required when the illuminant is a blackbody (cd /m^2)
 
	Returns:
		il: An illuminant structure
	"""
	def __init__(self, illuminant_name, wave = np.arange(400, 701, 10)):
		# Read in the wavelength and energy data from the illuminant file and extrapolate to intended wavelength
		illuminant_file = self.illuminant_read(illuminant_name)
		energy = read_spectral(illuminant_file, wave)
  
		# Set the wavelength and energy
		self.photons = energy_to_quanta(wave, energy)
		self.energy = energy
		self.wave = wave
		
		# Adjust the luminance of the illuminant to the target
		self.adjust_luminance(100)
  
	def illuminant_read(self, illuminant_name):
		"""
		Reads in the illuminant data from the file directory by mapping illuminant name to the illuminant file
  
		Args:
			illuminant_name (str): Short and simple name for the illuminant

		Returns:
			illuminant_file (str): Full name of the illuminant in the directory 
		"""
		# Format the illuminant name into lower case and removing blank spaces
		illuminant_name = illuminant_name.replace(" ", "").lower()
		
		base_dir = 'data/lights/'
		illuminant_paths = {	
			'd50': 'D50.mat',
			'd55': 'D55.mat',
			'd65': 'D65.mat',
			'd75': 'D75.mat',
			'fluorescent': 'Fluorescent.mat',
			'illuminanta': 'illuminantA.mat',
			'illuminantb': 'illuminantB.mat',
			'illuminantc': 'illuminantC.mat',
			'led1839': 'LED_1839.mat',
			'led2977': 'LED_2977.mat',
			'led3150': 'LED_3150.mat',
			'led3845': 'LED_3845.mat',
			'led4244': 'LED_4244.mat',
			'led4613': 'LED4613.mat',
			'led400': 'LED400.mat',
			'led405': 'LED405.mat',
			'led425': 'LED425.mat',
			'led450': 'LED450.mat',
			'tungsten': 'Tungsten.mat'			
		}
  
		try:
			return f'{base_dir}{illuminant_paths[illuminant_name]}'
		except KeyError:
			raise ValueError(f'Unknown illuminant name: {illuminant_name}')

	def adjust_luminance(self, target_luminance):
		# Calculate the current luminance from energy and wave
		current_luminance = luminance_from_energy(self.energy, self.wave)
  
		# Adjust the energy based on target luminance and current luminance
		new_energy = self.energy * (target_luminance / current_luminance.mean())
  
        # Update photons and energy to the new luminance
		self.photons = energy_to_quanta(self.wave, new_energy)
		self.energy = new_energy
        
		return

if __name__ == '__main__':
    illuminant = Illuminant('d65')
    print(illuminant.photons)
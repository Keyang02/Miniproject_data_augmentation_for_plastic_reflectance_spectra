import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from material_dict import material_labels, noise_labels, color_labels

class Dataset_csv(Dataset):
    def __init__(self, dataset_path, print_info=False):
        self.dataset, self.labels = self.build_dataset(dataset_path, print_info)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        spectrum = torch.unsqueeze(spectrum, 0)
        label = self.labels[idx]
        return spectrum, label

    def build_dataset(self, dataset_path, print_info):
        '''get dataset of signal'''
        df = pd.read_csv(dataset_path)
        if print_info:
            print(f"The number of each material: {df['Material'].value_counts()}")
        labels = df['Material'].values.copy()
        spectra = df.drop(columns=['Material']).values
        labels = torch.tensor(labels, dtype=torch.int)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        return spectra, labels
    
class Dataset_addnoise(Dataset):
    def __init__(self, dataset_path, print_info=False, 
                 wmin=400, wmax=1700, n_wavelengths=1301,
                 relative_uncertainty=0.005,
                 deformation_parameters=(1, 10, -20, 20),
                 colorslopes_path='PlasticDataset/color/dye_extinctioncoeff_slopes.csv'
                 ):
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.relative_uncertainty = relative_uncertainty  # Relative uncertainty for Gaussian noise
        # Parameters for thermal deformation: (slope(cm^-1/K), parameter to calculate FWHM, min_temp_change(K), max_temp_change(K))
        self.deformation_parameters = deformation_parameters
        self.colorslopes_path = colorslopes_path

        self.dataset, self.labels, self.noisetype = self.build_dataset(dataset_path, print_info)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        material = self.labels[idx]
        noisetype = self.noisetype[idx]

        # if not self.isexpand:
        spectrum, noisetype = self.addnoise(spectrum, noisetype)
        spectrum = torch.unsqueeze(spectrum, 0)        # Add channel dimension

        return spectrum, material, noisetype

    def build_dataset(self, dataset_path, print_info):
        '''get dataset of signal'''
        df = pd.read_csv(dataset_path)
        if print_info:
            print(f"The number of each material: {df['Material'].value_counts()}")
        material = df['Material'].values.copy()
        spectra = df.drop(columns=['Material']).values
        material = torch.tensor(material, dtype=torch.int)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        noisetype = torch.tensor(np.random.choice([0, 1, 2, 3], size=len(material)), dtype=torch.int)
        return spectra, material, noisetype
    
    def addnoise(self, spectrum, noisetype):
        '''add noise to the signal'''
        match noisetype.item():
            case 0:
                # Ideal case, no noise added
                return spectrum, noisetype
            case 1:
                # Gaussian noise
                Gaussian_noise = torch.randn_like(spectrum, dtype=torch.float32) * self.relative_uncertainty * spectrum
                spectrum = torch.clamp(spectrum + Gaussian_noise, 0, 1)  
                return spectrum, noisetype
            case 2:
                # Distortion noise
                spectrum = self.thermal_deformation(spectrum, random.uniform(-20, 20))
                return spectrum, noisetype
            case 3:
                # Color 
                spectrum, is_white = self.addcolor(spectrum)
                if not is_white:
                    noisetype = torch.tensor(0, dtype=torch.int)
                return spectrum, noisetype

    def thermal_deformation(self, spectrum, temperature_change):
        spectrum = spectrum.detach().numpy()
        wavelengths = self.wavelengths

        slope = self.deformation_parameters[0]  # Slope in cm^-1/K
        fwhm = self.deformation_parameters[1]  # Parameter to calculate FWHM in cm^-1
        wavenumbers_wl = 1 / (wavelengths * 1e-7)  # Convert nm to cm^-1
        # Sort wavenumbers and spectrum to ensure they are ordered from low to high
        idx = np.argsort(wavenumbers_wl)
        wavenumbers_wl = wavenumbers_wl[idx]
        spectrum = spectrum[idx]

        wavenumbers_uniform = np.linspace(wavenumbers_wl[0], wavenumbers_wl[-1], len(wavenumbers_wl))
        spectrum_wn = np.interp(wavenumbers_uniform, wavenumbers_wl, spectrum)

        dwn = wavenumbers_uniform[1] - wavenumbers_uniform[0]  # Wavenumber resolution

        # Calculate the wavelength drift caused by temperature change
        wavenumber_drift = wavenumbers_uniform  + slope * temperature_change
        spectrum_wn = np.interp(wavenumber_drift, wavenumbers_uniform, spectrum_wn)
        # Apply Gaussian filter to simulate thermal deformation

        sig = fwhm * np.sqrt(abs(temperature_change) / 2 / np.log(2))
        kernel_size = int(np.ceil(4 * sig / dwn)) + 1

        if kernel_size <= 1:
            x = np.linspace(-kernel_size * dwn / 2, kernel_size * dwn / 2, kernel_size)
            kernel = np.exp(-0.5 * (x / sig) ** 2)
            kernel = kernel / np.sum(kernel)  # Normalize the kernel

            pad_left = kernel.size // 2
            pad_right = kernel.size - pad_left - 1
            spectrum_wn = np.pad(spectrum_wn, (pad_left, pad_right), mode='edge')
            spectrum_wn = np.convolve(spectrum_wn, kernel, mode='valid')

        spectrum = np.interp(wavenumbers_wl, wavenumbers_uniform, spectrum_wn)
        spectrum = spectrum[idx]
        spectrum = np.clip(spectrum, 0, 1)  # Ensure reflectance is within [0, 1]
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        return spectrum

    def addcolor(self, spectrum, color = 'random', relative_concentration = (0.5, 1)):
        spectrum_original = spectrum.clone()
        if (max(spectrum) > 0.6) & (max(spectrum) < 1.0):
            spectrum = spectrum.detach().numpy()
            wavelengths = self.wavelengths
            # Read the slope of extinction coefficients over concentration for the specified color
            if color == 'random':
                color = random.choice(list(color_labels.values()))
            k_slopes = pd.read_csv(self.colorslopes_path)
            k_slope = k_slopes[color].to_numpy(dtype=float)
            # Calculate refractive index based on reflectance spectrum
            fraction = 1 / (1 - spectrum)
            if (fraction < 0).any() or np.isnan(fraction).any():
                return spectrum_original, False
            n = fraction + np.sqrt(fraction ** 2 - 1)
            # Generate random concentration within the specified range
            a = np.random.uniform(relative_concentration[0], relative_concentration[1])
            # Varation of the reflectance
            dr = - a * k_slope * 2 * np.pi / wavelengths * ((n ** 2 - 1) / (n ** 2 + 1)) ** 2
            spectrum = spectrum + dr
            spectrum = np.clip(spectrum, 0, 1)  # Ensure reflectance is within [0, 1]
            return torch.tensor(spectrum, dtype=torch.float32), True
        else:
            return spectrum_original, False  # Return original spectrum if it is already colored
        
class Dataset_expand(Dataset):
    def __init__(self, dataset_path, print_info=False, 
                 wmin=400, wmax=1700, n_wavelengths=1301,
                 expand_factor:list | bool=(0.2, 0.2, 0.2),                 # (Gaussian, Distortion, Color)
                 relative_uncertainty=0.005,
                 deformation_parameters=(1, 10, -20, 20),
                 colorslopes_path='PlasticDataset/color/dye_extinctioncoeff_slopes.csv'
                 ):
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.relative_uncertainty = relative_uncertainty  # Relative uncertainty for Gaussian noise
        # Parameters for thermal deformation: (slope(cm^-1/K), parameter to calculate FWHM, min_temp_change(K), max_temp_change(K))
        self.deformation_parameters = deformation_parameters
        self.colorslopes_path = colorslopes_path

        self.dataset, self.labels, self.noisetype = self.build_dataset(dataset_path, print_info)

        if type(expand_factor) == list:
            self.expand_dataset(expand_factor)
        elif expand_factor:
            self.expand_dataset([0.2, 0.2, 0.2])  # Default expansion factors if not specified
        else:
            print("No expansion applied to the dataset.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        material = self.labels[idx]
        noisetype = self.noisetype[idx]
        spectrum = torch.unsqueeze(spectrum, 0)        # Add channel dimension
        return spectrum, material, noisetype

    def build_dataset(self, dataset_path, print_info):
        '''get dataset of signal'''
        df = pd.read_csv(dataset_path)
        if print_info:
            print(f"The number of each material: {df['Material'].value_counts()}")
        material = df['Material'].values.copy()
        spectra = df.drop(columns=['Material']).values
        material = torch.tensor(material, dtype=torch.int)          # Material labels
        spectra = torch.tensor(spectra, dtype=torch.float32)        # Spectra values
        noisetype = torch.zeros(len(material), dtype=torch.int)     # Initialize noisetype with zeros
        return spectra, material, noisetype
    
    def expand_dataset(self, expand_factor):
        '''expand dataset by adding noise'''
        n_gaussian = int(expand_factor[0] * self.__len__())
        n_distortion = int(expand_factor[1] * self.__len__())
        n_color = int(expand_factor[2] * self.__len__())

        noise_list = [1, 2, 3]  # Gaussian, Distortion, Color
        n_list = [n_gaussian, n_distortion, n_color]

        for i in range(len(noise_list)):
            for _ in range(n_list[i]):
                random_idx = random.randint(0, len(self.dataset) - 1)
                spectrum, material, _ = self.__getitem__(random_idx)
                spectrum_gaussian, noisetype = self.addnoise(spectrum.view(-1), noise_list[i])
                self.dataset = torch.cat((self.dataset, spectrum_gaussian.unsqueeze(0)), dim=0)
                self.labels = torch.cat((self.labels, material.unsqueeze(0)), dim=0)
                self.noisetype = torch.cat((self.noisetype, noisetype.unsqueeze(0)), dim=0)

    def addnoise(self, spectrum, noisetype: int):
        '''add noise to the signal'''
        match noisetype:
            case 1:
                # Gaussian noise
                Gaussian_noise = torch.randn_like(spectrum, dtype=torch.float32) * self.relative_uncertainty * spectrum
                spectrum = torch.clamp(spectrum + Gaussian_noise, 0, 1)
                noisetype = torch.tensor(1, dtype=torch.int)
                return spectrum, noisetype
            case 2:
                # Distortion noise
                spectrum = self.thermal_deformation(spectrum, random.uniform(-20, 20))
                noisetype = torch.tensor(2, dtype=torch.int)
                return spectrum, noisetype
            case 3:
                # Color 
                spectrum, is_white = self.addcolor(spectrum)
                if not is_white:
                    noisetype = torch.tensor(0, dtype=torch.int)
                else:
                    noisetype = torch.tensor(3, dtype=torch.int)
                return spectrum, noisetype

    def thermal_deformation(self, spectrum, temperature_change):
        spectrum = spectrum.detach().numpy()
        wavelengths = self.wavelengths

        slope = self.deformation_parameters[0]  # Slope in cm^-1/K
        fwhm = self.deformation_parameters[1]  # Parameter to calculate FWHM in cm^-1
        wavenumbers_wl = 1 / (wavelengths * 1e-7)  # Convert nm to cm^-1
        # Sort wavenumbers and spectrum to ensure they are ordered from low to high
        idx = np.argsort(wavenumbers_wl)
        wavenumbers_wl = wavenumbers_wl[idx]
        spectrum = spectrum[idx]

        wavenumbers_uniform = np.linspace(wavenumbers_wl[0], wavenumbers_wl[-1], len(wavenumbers_wl))
        spectrum_wn = np.interp(wavenumbers_uniform, wavenumbers_wl, spectrum)

        dwn = wavenumbers_uniform[1] - wavenumbers_uniform[0]  # Wavenumber resolution

        # Calculate the wavelength drift caused by temperature change
        wavenumber_drift = wavenumbers_uniform  + slope * temperature_change
        spectrum_wn = np.interp(wavenumber_drift, wavenumbers_uniform, spectrum_wn)
        # Apply Gaussian filter to simulate thermal deformation

        sig = fwhm * np.sqrt(abs(temperature_change) / 2 / np.log(2))
        kernel_size = int(np.ceil(4 * sig / dwn)) + 1

        if kernel_size <= 1:
            x = np.linspace(-kernel_size * dwn / 2, kernel_size * dwn / 2, kernel_size)
            kernel = np.exp(-0.5 * (x / sig) ** 2)
            kernel = kernel / np.sum(kernel)  # Normalize the kernel

            pad_left = kernel.size // 2
            pad_right = kernel.size - pad_left - 1
            spectrum_wn = np.pad(spectrum_wn, (pad_left, pad_right), mode='edge')
            spectrum_wn = np.convolve(spectrum_wn, kernel, mode='valid')

        spectrum = np.interp(wavenumbers_wl, wavenumbers_uniform, spectrum_wn)
        spectrum = spectrum[idx]
        spectrum = np.clip(spectrum, 0, 1)  # Ensure reflectance is within [0, 1]
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        return spectrum

    def addcolor(self, spectrum, color = 'random', relative_concentration = (0.2, 0.5)):
        spectrum_original = spectrum.clone()
        if (max(spectrum) > 0.6) & (max(spectrum) < 1.0):
            spectrum = spectrum.detach().numpy()
            wavelengths = self.wavelengths
            # Read the slope of extinction coefficients over concentration for the specified color
            if color == 'random':
                color = random.choice(list(color_labels.values()))
            k_slopes = pd.read_csv(self.colorslopes_path)
            k_slope = k_slopes[color].to_numpy(dtype=float)
            # Calculate refractive index based on reflectance spectrum
            fraction = 1 / (1 - spectrum)
            if (fraction < 0).any() or np.isnan(fraction).any():
                return spectrum_original, False
            n = fraction + np.sqrt(fraction ** 2 - 1)
            # Generate random concentration within the specified range
            a = np.random.uniform(relative_concentration[0], relative_concentration[1])
            # Varation of the reflectance
            dr = - a * k_slope * 2 * np.pi / wavelengths * ((n ** 2 - 1) / (n ** 2 + 1)) ** 2
            spectrum = spectrum + dr
            spectrum = np.clip(spectrum, 0, 1)  # Ensure reflectance is within [0, 1]
            return torch.tensor(spectrum, dtype=torch.float32), True
        else:
            return spectrum_original, False  # Return original spectrum if it is already colored


if __name__ == '__main__':
    dataset = Dataset_addnoise('PlasticDataset/labeled/merged_dataset_withlabel.csv')
    print(f"Dataset length: {len(dataset)}")
    datapoint, label, noisetype = dataset.__getitem__(0)
    print(f"First datapoint shape: {datapoint.shape}")
    print(f"First label: {label}")
    print(f"First noisetype: {noisetype}")
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True
    )

    for _, (spectrum, label, noisetype) in enumerate(trainloader):
        if noisetype.item() == 1:
            plt.plot(spectrum.detach().numpy().flatten())
            plt.xlabel("Wavelength")
            plt.ylabel("Reflectance")
            plt.savefig(f"img/spectrum_{material_labels[label.item()]}_noisetype_{noise_labels[noisetype.item()]}.png")
            plt.close()
            break

    for _, (spectrum, label, noisetype) in enumerate(trainloader):
        if noisetype.item() == 2:
            plt.plot(spectrum.detach().numpy().flatten())
            plt.xlabel("Wavelength")
            plt.ylabel("Reflectance")
            plt.savefig(f"img/spectrum_{material_labels[label.item()]}_noisetype_{noise_labels[noisetype.item()]}.png")
            plt.close()
            break

    for _, (spectrum, label, noisetype) in enumerate(trainloader):
        if noisetype.item() == 3:
            plt.plot(spectrum.detach().numpy().flatten())
            plt.xlabel("Wavelength")
            plt.ylabel("Reflectance")
            plt.savefig(f"img/spectrum_{material_labels[label.item()]}_noisetype_{noise_labels[noisetype.item()]}.png")
            plt.close()
            break

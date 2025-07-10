import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from material_dict import material_labels, noise_labels, color_labels

class Dataset_csv(Dataset):
    def __init__(self, root, print_info=False):
        self.dataset, self.labels = self.build_dataset(root, print_info)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        spectrum = torch.unsqueeze(spectrum, 0)
        label = self.labels[idx]
        return spectrum, label

    def build_dataset(self, root, print_info):
        '''get dataset of signal'''
        df = pd.read_csv(root)
        if print_info:
            print(f"The number of each material: {df['Material'].value_counts()}")
        labels = df['Material'].values.copy()
        spectra = df.drop(columns=['Material']).values
        labels = torch.tensor(labels, dtype=torch.int)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        return spectra, labels
    
class Dataset_addnoise(Dataset):
    def __init__(self, root, print_info=False, wmin=400, wmax=1700, n_wavelengths=1301):
        self.dataset, self.labels = self.build_dataset(root, print_info)
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.gaussian_amplitude = 0.0005  # Amplitude for Gaussian noise
        self.distortion_factor = 0.005  # Distortion factor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        material = self.labels[idx]
        noisetype = random.choices(
            [0, 1, 2, 3], 
            weights=[0.4, 0.2, 0.2, 0.2], 
        )
        noisetype = torch.tensor(noisetype, dtype=torch.int)
        if material.item() != 0:
            spectrum = self.addnoise(spectrum, noisetype)
        spectrum = torch.unsqueeze(spectrum, 0)        # Add channel dimension
        return spectrum, material, noisetype

    def build_dataset(self, root, print_info):
        '''get dataset of signal'''
        df = pd.read_csv(root)
        if print_info:
            print(f"The number of each material: {df['Material'].value_counts()}")
        material = df['Material'].values.copy()
        spectra = df.drop(columns=['Material']).values
        material = torch.tensor(material, dtype=torch.int)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        return spectra, material
    
    def addnoise(self, spectrum, noisetype):
        '''add noise to the signal'''
        match noisetype.item():
            case 0:
                # Ideal case, no noise added
                return spectrum
            case 1:
                # Gaussian noise
                Gaussian_noise = torch.randn_like(spectrum) * self.gaussian_amplitude
                spectrum = torch.clamp(spectrum + Gaussian_noise, 0, 1)
                return spectrum
            case 2:
                # Distortion noise
                wavelengths = self.wavelengths 
                wavelengths_drift = wavelengths + np.random.normal(0, self.distortion_factor, wavelengths.shape)  # Apply horizontal random shifts
                wavelengths_drift = np.sort(wavelengths_drift)  # Sort to maintain order
                wavelengths_drift[0] = 400  # Ensure the first wavelength is fixed
                wavelengths_drift[-1] = 1700  # Ensure the last wavelength is fixed
                spectrum = torch.tensor(np.interp(wavelengths_drift, wavelengths, spectrum.numpy()))
                return spectrum
            case 3:
                # Color 
                return self.addcolor(spectrum)

    def addcolor(self, spectrum, color = 'random', relative_concentration = (0.5, 1)):
        spectrum = spectrum.detach().numpy()
        wavelengths = self.wavelengths
        # Read the slope of extinction coefficients over concentration for the specified color
        if color == 'random':
            color = random.choice(list(color_labels.values()))
        k_slopes = pd.read_csv('PlasticDataset/color/dye_extinctioncoeff_slopes.csv')        
        k_slope = k_slopes[color].to_numpy(dtype=float)
        # Calculate refractive index based on reflectance spectrum
        fraction = 1 / (1 - spectrum)
        n = fraction + np.sqrt(fraction ** 2 - 1)
        # Generate random concentration within the specified range
        a = np.random.uniform(relative_concentration[0], relative_concentration[1])
        # Varation of the reflectance
        dr = - a * k_slope * 2 * np.pi / wavelengths * ((n ** 2 - 1) / (n ** 2 + 1)) ** 2
        spectrum = spectrum + dr
        spectrum = np.clip(spectrum, 0, 1)  # Ensure reflectance is within [0, 1]
        return torch.tensor(spectrum, dtype=torch.float32)

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

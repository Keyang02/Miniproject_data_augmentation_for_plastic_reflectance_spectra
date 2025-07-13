import torch
import torch.nn as nn

import os
import pandas as pd
import numpy as np

from conditional_WGAN_net import CondGen1D_ConvT_2labels as CondGen1D 
from material_dict import material_labels as material_dict

data_number_per_material = 300
nz = 100  

root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'
save_path = 'synthetic_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = CondGen1D(nz).to(device)
netG.load_state_dict(torch.load(os.path.join(root, 'nets/netG_epoch_100.pth')))

synthetic_data = pd.DataFrame()
columns = list(np.linspace(400, 1700, 1301))  # Assuming wavelengths from 400 to 1700 nm
columns = ['Material', 'NoiseType'] + columns
synthetic_data = pd.DataFrame(columns=columns)

for key in material_dict.keys():
    material_label = torch.tensor([key], device=device, dtype=torch.int)
    for i in range(data_number_per_material):
        noise_type = i % 4
        match noise_type:
            case 0:
                noise_label = torch.tensor([0], device=device, dtype=torch.int)
            case 1:
                noise_label = torch.tensor([1], device=device, dtype=torch.int)
            case 2:
                noise_label = torch.tensor([2], device=device, dtype=torch.int)
            case 3:
                noise_label = torch.tensor([3], device=device, dtype=torch.int)
        noise = torch.randn(1, nz, device=device)
        synthetic_spectra = netG(noise, material_label, noise_label)
        synthetic_spectra = synthetic_spectra.cpu().detach().flatten().numpy().tolist()
        row = [key, noise_type] + synthetic_spectra
        synthetic_data.loc[len(synthetic_data)] = row
        print(f' {i + 1} / {data_number_per_material} synthetic spectra generated for {material_dict[key]}', end='\r')

synthetic_data.to_csv(os.path.join(root, save_path, 'synthetic_data.csv'), index=False)
print(f'\nSynthetic data generated and saved to {os.path.join(root, save_path, "synthetic_data.csv")}')

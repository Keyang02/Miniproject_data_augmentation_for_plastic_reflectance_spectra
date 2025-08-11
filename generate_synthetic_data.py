import torch

import os
import argparse
import pandas as pd
import numpy as np

from typing import Literal

from conditional_WGAN_net import CondGen1D_ConvT_2labels as CondGen1D 
from material_dict import material_labels as material_dict

parser = argparse.ArgumentParser(description='Generate Synthetic Data')
parser.add_argument('--data_number_per_material', type=int, default=1500, help='number of synthetic data per material')
parser.add_argument('--model_info', type=str, default='_k4_GP', help='information about the model')

allowable_noise_types = Literal['average', 'no_noise', 'emphasis_no_noise']

def get_noise_labels(data_number_per_material, device,
                   noise_type: allowable_noise_types):
    if noise_type == 'no_noise':
        noise_labels = torch.zeros(data_number_per_material, device=device, dtype=torch.int)

    elif noise_type == 'average':
        n_one_noise = data_number_per_material // 4
        noise_labels = torch.zeros(data_number_per_material, device=device, dtype=torch.int)
        noise_labels[n_one_noise:2*n_one_noise] = 1
        noise_labels[2*n_one_noise:3*n_one_noise] = 2
        noise_labels[3*n_one_noise:] = 3

    elif noise_type == 'emphasis_no_noise':
        n_one_noise = data_number_per_material // 8
        noise_labels = torch.zeros(data_number_per_material, device=device, dtype=torch.int)
        noise_labels[n_one_noise:2*n_one_noise] = 1
        noise_labels[2*n_one_noise:3*n_one_noise] = 2
        noise_labels[3*n_one_noise:] = 3

    return noise_labels

root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'
nz = 100  

data_number_per_material = parser.parse_args().data_number_per_material
net_G_path = f'nets/netG_con_wgan{parser.parse_args().model_info}.pth'
save_path = 'synthetic_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = CondGen1D(nz).to(device)
netG.load_state_dict(torch.load(os.path.join(root, net_G_path)))
synthetic_data = pd.DataFrame()
columns = list(np.linspace(400.0, 1700.0, 1301))  # Assuming wavelengths from 400 to 1700 nm
columns = ['Material', 'NoiseType'] + columns
synthetic_data = pd.DataFrame(columns=columns)

for key in material_dict.keys():
    material_label = torch.tensor([key], device=device, dtype=torch.int)
    noise_labels = get_noise_labels(data_number_per_material, device, 'emphasis_no_noise')
    for i in range(data_number_per_material):
        noise_label = noise_labels[i]
        noise = torch.randn(1, nz, device=device)
        synthetic_spectra = netG(noise, material_label, noise_label.unsqueeze(0))
        synthetic_spectra = synthetic_spectra.cpu().detach().flatten().numpy().tolist()
        row = [key, noise_label.item()] + synthetic_spectra
        synthetic_data.loc[len(synthetic_data)] = row
        print(f' {i + 1} / {data_number_per_material} synthetic spectra generated for {material_dict[key]}', end='\r')

synthetic_data.to_csv(os.path.join(root, save_path, 'synthetic_data.csv'), index=False)
print(f'\nSynthetic data generated and saved to {os.path.join(root, save_path, "synthetic_data.csv")}')

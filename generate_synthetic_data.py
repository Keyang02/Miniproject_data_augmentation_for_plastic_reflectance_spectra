import torch

import os
import argparse
import pandas as pd
import numpy as np

from typing import Literal

from conditional_WGAN_net import CondGen1D_Upsample_2labels as CondGen1D 
from material_dict import material_labels as material_dict

parser = argparse.ArgumentParser(description='Generate Synthetic Data')
parser.add_argument('--data_number_per_material', type=int, default=1500, help='number of synthetic data per material')
parser.add_argument('--model_info', type=str, default='_k4_GP', help='information about the model')
parser.add_argument('--selected_material', default=None, help='which material to use')

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

nz = 100
batch_size = 50

data_number_per_material = parser.parse_args().data_number_per_material
if parser.parse_args().selected_material is not None:
    if int(parser.parse_args().selected_material) in material_dict:
        net_G_path = f'nets/netG_{material_dict[int(parser.parse_args().selected_material)]}.pth'
    else:
        raise ValueError(f"Selected material {parser.parse_args().selected_material} is not in the material dictionary.")
else:
    net_G_path = f'nets/netG_con_wgan{parser.parse_args().model_info}.pth'
save_path = 'synthetic_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = CondGen1D(nz).to(device)
netG.load_state_dict(torch.load(net_G_path))
netG.eval()

synthetic_data = pd.DataFrame()
columns = list(np.linspace(400.0, 1700.0, 1301))  # Assuming wavelengths from 400 to 1700 nm
columns = ['Material', 'NoiseType'] + columns
synthetic_data = pd.DataFrame(columns=columns)

iteration_num = data_number_per_material // batch_size
total_rows = len(material_dict) * iteration_num * batch_size
out = np.empty((total_rows, 2 + 1301), dtype=np.float32)

row_ptr = 0
for key in material_dict.keys():
    material_label = torch.tensor([key], device=device, dtype=torch.int)
    material_label = material_label.repeat(batch_size)
    noise_labels = get_noise_labels(iteration_num, device, 'emphasis_no_noise')
    for i in range(iteration_num):
        noise_label = noise_labels[i]
        noise_label = noise_label.repeat(batch_size)
        noise = torch.randn(batch_size, nz, device=device)
        synthetic_spectra = netG(noise, material_label, noise_label)
        synthetic_spectra = torch.flatten(synthetic_spectra, start_dim=1)
        keys = [key] * batch_size
        keys = torch.tensor(keys, device=device, dtype=torch.int)
        block = torch.cat((keys.unsqueeze(1), noise_label.unsqueeze(1), synthetic_spectra), dim=1)

        block = block.cpu().detach().numpy()
        out[row_ptr:row_ptr + batch_size] = block
        row_ptr += batch_size

        print(f' {i + 1} / {iteration_num} synthetic spectra generated for {material_dict[key]}', end='\r', flush=True)

df = pd.DataFrame(out, columns=columns)
df[['Material', 'NoiseType']] = df[['Material', 'NoiseType']].astype(np.int32)

df.to_csv(os.path.join(save_path, 'synthetic_data.csv'), index=False)
print(f'\nSynthetic data generated and saved to {os.path.join(save_path, "synthetic_data.csv")}')

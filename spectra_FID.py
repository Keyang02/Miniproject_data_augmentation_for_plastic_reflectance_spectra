import os
from pathlib import Path
import base64
import argparse

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import linalg
from torch.utils.data import DataLoader, Dataset

from material_dict import material_labels

# -----------------------------------------------------------------------------
# 1. Data ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

spectrum_length = 1301  # sampling number of each spectrum

parser = argparse.ArgumentParser(description='Spectral FID Calculation')
parser.add_argument('--model_info', type=str, default='_k4_GP', help='information about the model')
parser.add_argument('--islog', type=bool, default=False, help='whether to log the training process')
parser.add_argument('--trainset', type=str, choices=['large', 'small'], default='large', help='which trainset to use')

log_path = f'log/spectral_fid_values{parser.parse_args().model_info}.md'



if parser.parse_args().trainset == 'small':
    trainset_path = 'PlasticDataset/small_sets/filtered_all_materials.csv'
    extractor_path = 'feature_extractor_CR_small.pth'
else:
    trainset_path = 'PlasticDataset/labeled_10materials/merged.csv'
    extractor_path = 'feature_extractor_CR.pth'

# Set random seeds for reproducibility
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DatasetReadCSV(Dataset):
    def __init__(self, 
                 path: str | Path, 
                 material_id: int = None,
                 wmin: float = 400, wmax: float = 1700, 
                 n_wavelengths: int = 1301,
                 ):
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.dataset, self.labels, self.noisetype = self.build_dataset(path, material_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        material = self.labels[idx]
        noisetype = self.noisetype[idx]
        spectrum = torch.unsqueeze(spectrum, 0)        # Add channel dimension
        return spectrum, material, noisetype

    def build_dataset(self, dataset_path, material_id):
        '''get dataset of signal'''
        df = pd.read_csv(dataset_path)
        if material_id is not None:
            df = df[df['Material'] == material_id]
        material = df['Material'].values.copy()

        if 'NoiseType' in df.columns:
            noisetype = df['NoiseType'].values.copy()
            noisetype = torch.tensor(noisetype, dtype=torch.int)
            df = df.drop(columns=['NoiseType'])
        else:
            noisetype = torch.zeros(len(material), dtype=torch.int)
        
        spectra = df.drop(columns=['Material']).values

        material = torch.tensor(material, dtype=torch.int)       # Material labels
        spectra = torch.tensor(spectra, dtype=torch.float32)        # Spectra values
        return spectra, material, noisetype

# -----------------------------------------------------------------------------
# 2. 1‑D CNN Encoder -----------------------------------------------------------
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.block(x)

def output_length_conv1D(input_len, kernel_size, stride=1, padding=0):
    """Calculate the output length of a 1D convolution."""
    return (input_len - kernel_size + 2 * padding) // stride + 1

class CondCritic1D_2labels(nn.Module):
    def __init__(self, 
                 embed_classes1: int = 11,
                 embed_dim1: int = 16, 
                 embed_classes2: int = 4,
                 embed_dim2: int = 8,
                 base_c: int = 64, 
                 feat_dim: int = 128):
        super().__init__()
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)

        self.blocks = nn.Sequential(
            nn.Conv1d(1, base_c, 4, stride=2, padding=1),   # L:1309 -> 654  C:1 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_c, base_c*2),                    # L:654 -> 327   C:64 -> 128
            ConvBlock(base_c*2, base_c*4),                  # L:327 -> 163   C:128 -> 256
            ConvBlock(base_c*4, base_c*8),                  # L:163 -> 81    C:256 -> 512
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_c*8, feat_dim, bias=False),
        )
        
    def forward(self, x, y1, y2):
        # x: (B,1,1301)   y: (B,)
        y1_emb = self.embed1(y1).unsqueeze(1)                    # (B,1,embed_dim)
        y2_emb = self.embed2(y2).unsqueeze(1)                    # (B,1,embed_dim)
        h = torch.cat([x, y1_emb, y2_emb], 2)                     # (B,1,1301+embed_dim*2)
        h = self.blocks(h)
        return h.view(h.size(0), -1)                                     # (B,1) Wasserstein score

# -----------------------------------------------------------------------------
# 3. Helpers -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_activations(loader: DataLoader, model: nn.Module, device: torch.device):
    """Collect encoder activations for all samples in *loader*."""
    model.eval()
    activations = []
    with torch.no_grad():
        for spectra, materials, noise in loader:  # DataLoader returns tuple
            spectra = spectra.to(device)
            materials = materials.to(device)
            noise = noise.to(device)
            activations.append(model(spectra, materials, noise).cpu().numpy())
    return np.concatenate(activations, axis=0)


def compute_statistics(acts: np.ndarray):
    """Return mean and covariance of activations."""
    mu = acts.mean(axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Classic Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    # covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    covmean = linalg.sqrtm(sigma1 @ sigma2)
    # Handle numerical errors (small imaginary parts)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if np.iscomplexobj(covmean):
        covmean = covmean.real  # numerical cleanup
    fid_val = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid_val)

# -----------------------------------------------------------------------------
# 4. Main ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def fed_one_material(material: int):  # noqa: D401
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_csv_path = trainset_path
    fake_csv_path = 'synthetic_data/synthetic_data.csv'
    # ------------------------ Data ------------------------
    real_ds = DatasetReadCSV(real_csv_path, material_id=material)
    fake_ds = DatasetReadCSV(fake_csv_path, material_id=material)

    if len(real_ds) == 0 or len(fake_ds) == 0:
        raise RuntimeError("One of the datasets is empty after filtering. Check material ID.")

    # Sanity‑check that spectral lengths match
    real_sample = real_ds[0][0]  # Get the first sample
    fake_sample = fake_ds[0][0]  # Get the first sample
    if fake_sample.shape[-1] != real_sample.shape[-1]:
        raise ValueError("Real and fake spectra sequence lengths do not match")

    real_loader = DataLoader(real_ds, batch_size=20, shuffle=False)
    fake_loader = DataLoader(fake_ds, batch_size=20, shuffle=False)

    # ------------------------ Model -----------------------
    encoder = CondCritic1D_2labels(
        embed_classes1=len(material_labels),
        embed_classes2=4,
        embed_dim1=16,
        embed_dim2=8,
        base_c=64,
        feat_dim=128
    )
    state = torch.load(extractor_path)
    encoder.load_state_dict(state, strict=False)  # Load state dict with strict=False to ignore missing keys
    encoder.to(device)

    # ------------------------ Compute FID -----------------
    acts_real = get_activations(real_loader, encoder, device)
    acts_fake = get_activations(fake_loader, encoder, device)

    mu_r, sig_r = compute_statistics(acts_real)
    mu_f, sig_f = compute_statistics(acts_fake)

    fid_value = calculate_fid(mu_r, sig_r, mu_f, sig_f)
    print(f"Spectral FID: {fid_value:.4f} for {material_labels.get(material, 'all materials')}")
    return fid_value

if __name__ == "__main__":
    fid_values = []
    for material_id in material_labels.keys():
        fid_value = fed_one_material(material_id)
        fid_values.append(fid_value)

    f, ax = plt.subplots(figsize=(10, 5))
    barchart = ax.bar(material_labels.values(), fid_values, color='skyblue')
    ax.bar_label(barchart, fmt='%.3f')
    ax.set_xlabel('Material')
    ax.set_ylabel('FID Value')
    ax.set_title('Spectral FID Values by Material')
    ax.legend(['FID'])
    plt.savefig('img/spectral_fid_values.png')

if parser.parse_args().islog:
    newline = f'{os.linesep}'
    materials = list(material_labels.values())
    FID_dict = dict(zip(materials, fid_values))

    with open('img/spectral_fid_values.png', 'rb') as f:
        img_data = f.read()
        img_data_b64 = base64.b64encode(img_data).decode('utf-8')
    with open(log_path, 'a') as md:
        md.write('> ### Spectral FID Values' + newline)
        md.write('> | Material | FID Value |' + newline)
        md.write('> |----------|-----------|' + newline)
        for material, fid in FID_dict.items():
            md.write(f'> | {material} | {fid:.4f} |' + newline)
        md.write(newline * 2)

        md.write('> ### FID Values for All Materials' + newline)
        md.write(f'> ![FID Values](data:image/png;base64,{img_data_b64})' + newline)
        md.write(newline * 2)

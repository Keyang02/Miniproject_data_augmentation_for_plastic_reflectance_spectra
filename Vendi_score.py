import numpy as np
import pandas as pd
import argparse
import base64
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns  # for nicer heatmaps
from typing import Union
from pathlib import Path

from material_dict import material_labels

parser = argparse.ArgumentParser(description='Vendi Score Calculation')
parser.add_argument('--embed_method', type=str, default='resample', choices=['resample', 'nn'], help='Method to embed spectra')
parser.add_argument('--model_info', type=str, default='_k4_CR', help='information about the model')
parser.add_argument('--vs_method', type=str, default='inner_product', choices=['inner_product', 'RBF'], help='Method to compute Vendi Score')
parser.add_argument('--islog', type=bool, default=True, help='Whether to log-transform the spectra')

if 'k4' in parser.parse_args().model_info:
    extractor_path = 'nets/feature_extractor_k4.pth'
elif 'k3' in parser.parse_args().model_info:
    extractor_path = 'nets/feature_extractor_k3.pth'
else:
    extractor_path = 'nets/feature_extractor.pth'

log_path = f'log/spectral_fid_values{parser.parse_args().model_info}.md'

class DatasetReadCSV(Dataset):
    def __init__(self, 
                 path: str | Path, 
                 material_id: int = None,
                 wmin: float = 400, wmax: float = 1700, 
                 n_wavelengths: int = 1301,
                 isresample: bool = True,
                 n_samples: int | None = 100
                 ):
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.dataset, self.labels, self.noisetype = self.build_dataset(path, material_id)
        if isresample:
            self.dataset = self.resample(n_samples)
            self.dataset = nn.functional.normalize(self.dataset, dim=1)  # Normalize spectra

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spectrum = self.dataset[idx, :]
        material = self.labels[idx]
        noisetype = self.noisetype[idx]
        spectrum = torch.unsqueeze(spectrum, 0)        # Add channel dimension
        return spectrum, material, noisetype

    def build_dataset(self, root, material_id):
        '''get dataset of signal'''
        df = pd.read_csv(root)
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

    def resample(self, n_samples):
        """Resample spectrum to the defined wavelengths."""
        new_wavelengths = np.linspace(self.wavelengths[0], self.wavelengths[-1], n_samples)
        old_spectra = self.dataset.numpy()
        new_spectra = np.zeros((len(self.dataset), n_samples))

        for i in range(len(old_spectra)):
            spectrum = old_spectra[i, :]
            new_spectrum = np.interp(new_wavelengths, self.wavelengths, spectrum)
            new_spectra[i, :] = new_spectrum

        return torch.tensor(new_spectra, dtype=torch.float32)

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
                 n_classes: int = 7, 
                 embed_dim1: int = 16,
                 embed_dim2: int = 8,
                 base_c: int = 64, 
                 feat_dim: int = 128):
        super().__init__()
        self.embed1 = nn.Embedding(n_classes, embed_dim1)
        self.embed2 = nn.Embedding(n_classes, embed_dim2)

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
        h = torch.cat([x, y1_emb, y2_emb], 2)                    # (B,1,1301+embed_dim*2)
        h = self.blocks(h)
        h = h.view(h.size(0), -1)                                     # (B,1) Wasserstein score
        h = nn.functional.normalize(h)
        return h

def vendi_score_inner_product(X: torch.Tensor) -> float:
    """
    Compute Vendi Score for an (n x d) feature matrix X.
    """
    n, d = X.shape
    # Compute Gram matrix
    K = X @ X.T                  # shape (n, n)
    K_normalized = K / n         # divide by n
    # Eigenvalues or directly trace of K_normalized log K_normalized
    # But since trace( A log A ) = sum_i λ_i log λ_i for eigenvalues λ_i of A:
    eigvals = torch.linalg.eigvalsh(K_normalized)
    # numerical safety: drop zero or negative due to float error
    eigvals = eigvals[eigvals > 0]
    entropy = - torch.sum(eigvals * torch.log(eigvals))
    VS = torch.exp(entropy)

    return VS.numpy(), K.numpy()

def vendi_score_RBF(X: torch.Tensor, gamma: float = 10.0) -> float:
    """
    Compute Vendi Score for an (n x d) feature matrix X.
    """
    n, d = X.shape
    X_norm_1 = (X**2).sum(dim=1, keepdim=True)
    X_norm_2 = (X**2).sum(dim=1, keepdim=True).T
    K = X_norm_1 + X_norm_2 - 2 * (X @ X.T)

    K = torch.exp(-gamma * K)  # RBF kernel
    K_normalized = K / n  # Normalize by n

    # Eigenvalues or directly trace of K_normalized log K_normalized
    eigvals = torch.linalg.eigvalsh(K_normalized)
    # numerical safety: drop zero or negative due to float error
    eigvals = eigvals[eigvals > 0]
    entropy = - torch.sum(eigvals * torch.log(eigvals))
    VS = torch.exp(entropy)

    return VS.numpy(), K.numpy()

vendi_score_methods = {
    'inner_product': vendi_score_inner_product,
    'RBF': vendi_score_RBF
}

def compute_vs(X: torch.Tensor, method: str = 'inner_product', **kwargs) -> float:
    if method in vendi_score_methods:
        return vendi_score_methods[method](X, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_vs_one_material(csv_path: Union[str, Path],
                        feature_extractor: torch.nn.Module | None,
                        material: int):
    """
    Reads CSV, extracts features, computes Vendi Scores by material,
    and plots bar chart + Gram matrix heatmaps.
    """
    # 1) Load entire dataset
    if parser.parse_args().embed_method == 'nn':
        isresample = False
    else:
        isresample = True
    dataset = DatasetReadCSV(csv_path, material_id=material, isresample=isresample)
    dataset_length = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_length, shuffle=False)
    
    # 2) For each material, extract features and compute VS
    if parser.parse_args().embed_method == 'nn':
        feature_extractor.eval()
        with torch.no_grad():
            spectra, material_type, noise_types = next(iter(dataloader))
            # assume feature_extractor takes (batch, 1, n_wl) and returns (batch, d)
            feats = feature_extractor(spectra, material_type, noise_types)
            VS, K = compute_vs(feats, method=parser.parse_args().vs_method)
    elif parser.parse_args().embed_method == 'resample':
        # Resample spectra to fixed wavelengths
        spectra, material_type, noise_types = next(iter(dataloader))
        spectra = spectra.flatten(start_dim=1)  # Flatten to (batch, n_wl)
        VS, K = compute_vs(spectra, method=parser.parse_args().vs_method)

    return VS, K

def plot_vs(vs_scores, kernel_matrices):
    # 3) Bar chart of VS
    mats = list(vs_scores.keys())
    scores = [vs_scores[m] for m in mats]
    f, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar([str(m) for m in mats], scores)
    ax.bar_label(bars, fmt='%.2f')
    ax.set_xticks(range(len(material_labels.keys())))
    ax.set_xticklabels([material_labels[m] for m in mats])
    ax.set_xlabel("Material ID")
    ax.set_ylabel("Vendi Score")
    ax.set_title("Vendi Score by Material")
    plt.tight_layout()
    plt.savefig("vs_img/vendi_scores.png")
    
    # 4) Heatmaps of K
    for m, K in kernel_matrices.items():
        plt.figure(figsize=(6,5))
        sns.heatmap(K, cmap='viridis')
        plt.title(f"Gram Matrix K for {material_labels[m]}")
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()
        plt.savefig(f"vs_img/gram_matrix_{material_labels[m]}.png")


if __name__ == "__main__":
    # extractor = CondCritic1D_2labels()  
    # state = torch.load('nets/netD_con_wgan_k4_CR.pth')
    # extractor.load_state_dict(state, strict=False)
    csv_file = "synthetic_data/synthetic_data.csv"

    vs_scores = {}
    kernel_matrices = {}

    for mat_id in material_labels.keys():
        VS, K = compute_vs_one_material(csv_file, None, mat_id)
        vs_scores[mat_id] = VS
        kernel_matrices[mat_id] = K
        print(f"Material {material_labels[mat_id]}: Vendi Score = {VS:.4f}")

    plot_vs(vs_scores, kernel_matrices)
    print("Vendi Scores and Gram matrices computed and saved.")

    if parser.parse_args().islog:
        newline = f'{os.linesep}'
        with open("vs_img/vendi_scores.png", 'rb') as f:
            img_data = f.read()
            img_data_b64 = base64.b64encode(img_data).decode('utf-8')
        with open(log_path, 'a') as md:
            md.write('> ### Vendi Scores' + newline)
            md.write(f'> ![Vendi Scores](data:image/png;base64,{img_data_b64})' + newline)
            md.write(newline * 2)
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import pandas as pd
import imageio
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from loaddataset import Dataset_expand
from material_dict import material_labels

from conditional_WGAN_net import init_weights
from conditional_WGAN_net import CondCritic1D_2labels as CondCritic1D
from conditional_WGAN_net import CondGen1D_Upsample_2labels as CondGen1D

color_labels = {
    1 : 'blue',
    2 : 'orange',
    3 : 'pink',
    4 : 'yellow'
}

parser = argparse.ArgumentParser(description='Hyperparameters for Conditional WGAN-GP')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
parser.add_argument('--lr_D', type=float, default=5e-5, help='learning rate for Adam optimizer (Discriminator)')
parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for Adam optimizer (Generator)')
parser.add_argument('--gp_coeff', type=float, default=10, help='gradient penalty coefficient')
parser.add_argument('--cl_coeff', type=float, default=5, help='consistency loss coefficient')
parser.add_argument('--n_critic', type=int, default=5, help='D updates per G update')
parser.add_argument('--nz', type=int, default=100, help='noise dimension')
parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
parser.add_argument('--model_info', type=str, default='_CR_small', help='information about the model')
parser.add_argument('--islog', type=bool, default=False, help='whether to log the training process')
parser.add_argument('--selected_material', type=int, default=4, help='which material to use')
parser.add_argument('--activation', type=str, default='tanh', help='activation function for the generator output')

material_labels = {parser.parse_args().selected_material: material_labels[parser.parse_args().selected_material]}

# hyperparameters
beta1 = parser.parse_args().beta1               # beta1 for Adam optimizer
beta2 = parser.parse_args().beta2               # beta2 for Adam optimizer
lr_D = parser.parse_args().lr_D                 # learning rate for Adam optimizer (Discriminator)
lr_G = parser.parse_args().lr_G                 # learning rate for Adam optimizer (Generator)
gp_coeff = parser.parse_args().gp_coeff         # gradient penalty coefficient
cl_coeff = parser.parse_args().cl_coeff         # consistency loss coefficient
n_critic = parser.parse_args().n_critic         # D updates per G update
nz = parser.parse_args().nz                     # noise dimension
epoch_num = parser.parse_args().epoch_num       # number of epochs to train
batch_size = parser.parse_args().batch_size     # batch size for training

suffix = parser.parse_args().model_info         # model information suffix for output files
imgname = 'wgan_gp_epoch_{epoch}.png'
net_G_path = f'nets/netG_{material_labels[parser.parse_args().selected_material]}.pth'
net_D_path = f'nets/netD_{material_labels[parser.parse_args().selected_material]}.pth'
log_path = f'log/spectral_fid_values{suffix}.md'
loss_path = f'log/loss{suffix}.md'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

material_num = len(material_labels)

def compute_gradient_penalty(D, real_samples, fake_samples, material_label, noise_label, oneside = False):
    """Calculates the gradient penalty loss for WGAN GP"""
    if gp_coeff == 0:
        return torch.tensor(0.0, device=device)
    batch_size = real_samples.size(0)
    # sample interpolation coefficients in [0,1]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    # create interpolations
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates, material_label, noise_label)
    # ones for grad_outputs
    ones = torch.ones(d_interpolates.size(), device=device)
    # compute gradients wrt interpolates
    grads = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    grad_deviation = grad_norm - 1

    if oneside:
        # If grad_norm is less than 1, the penalty is zero.
        grad_deviation = torch.where(grad_deviation < 0, torch.zeros_like(grad_deviation), grad_deviation)

    penalty = gp_coeff * (grad_deviation ** 2).mean()
    return penalty

def intermediate_plot(fake, fixed_labels, imgname, material_num=11):
    f, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    for rowidx in range(2):
        for colidx in range(2):
            idx = rowidx * 2 + colidx
            axes[rowidx][colidx].plot(fake[idx].flatten().numpy(), linewidth=2.5)
            axes[rowidx][colidx].set_xticks([])
            axes[rowidx][colidx].set_yticks([])
            for spine in axes[rowidx][colidx].spines.values():
                spine.set_linewidth(2)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(os.path.join('img', imgname))
    plt.close(f)

class DataAugmentation:
    def __init__(self, 
                 wmin=400, wmax=1700, n_wavelengths=1301,
                 relative_uncertainty=0.005,
                 deformation_parameters=(1, 10),
                 colorslopes_path='PlasticDataset/color/dye_extinctioncoeff_slopes.csv'):
        '''Initialize data augmentation parameters.'''
        self.wavelengths = np.linspace(wmin, wmax, n_wavelengths)
        self.relative_uncertainty = relative_uncertainty
        self.deformation_parameters = deformation_parameters
        self.k_slopes_df = pd.read_csv(colorslopes_path)

    def augment(self, spectra_batch):
        '''Apply data augmentation to a batch of spectra.'''
        augmented = []
        noise_types = []
        # no autograd needed for augmentation itself
        with torch.no_grad():
            for idx in range(spectra_batch.shape[0]):
                noisetype = random.choices([1, 2, 3], weights=[0.2, 0.2, 0])[0]  # color noise
                one_spectrum = spectra_batch[idx, 0, :]

                spectrum, nt = self.addnoise(one_spectrum, noisetype)
                if nt.item() == 0:  # retry color failure
                    noisetype = random.choice([1, 2])
                    spectrum, nt = self.addnoise(one_spectrum, noisetype)

                spectrum = spectrum.unsqueeze(0).to(device)
                augmented.append(spectrum)
                noise_types.append(nt)

        augmented_spectra = torch.stack(augmented, dim=0).to(device)
        noise_batch = torch.stack(noise_types, dim=0).to(device, dtype=torch.int)
        return augmented_spectra, noise_batch

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
                spectrum = self.thermal_deformation(spectrum, random.uniform(0, 20))
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
        spectrum = spectrum.detach().cpu().numpy()
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
        wavenumber_drift = wavenumbers_uniform  - slope * temperature_change
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

    def addcolor(self, spectrum, color='random', relative_concentration=(0.2, 0.4)):
        # Keep original around to return if we skip
        spectrum_original = spectrum

        # Fast max check (no Python built-in max)
        smax = spectrum.max().item()
        if not (0.6 < smax < 1.0):
            return spectrum_original, False

        # Choose color
        if color == 'random':
            color = random.choice(list(color_labels.values()))

        # Move constants to the same device/dtype as spectrum
        device = spectrum.device
        dtype = spectrum.dtype
        wavelengths = torch.as_tensor(self.wavelengths, device=device, dtype=dtype)

        # k_slope as torch tensor (column from cached DF)
        k_slope_np = self.k_slopes_df[color].to_numpy(dtype=float)
        k_slope = torch.as_tensor(k_slope_np, device=device, dtype=dtype)

        # Reflectance → refractive index (avoid NaNs/inf with protective clamp)
        # fraction = 1 / (1 - R)
        R = spectrum.clamp(max=0.9999)
        fraction = 1.0 / (1.0 - R)

        # Guard invalid: if any non-finite shows up, skip
        if not torch.isfinite(fraction).all() or (fraction < 0).any():
            return spectrum_original, False

        n = fraction + torch.sqrt(fraction * fraction - 1.0)

        # Random concentration a ~ U(low, high)
        a = (relative_concentration[0] +
            (relative_concentration[1] - relative_concentration[0]) *
            torch.rand((), device=device, dtype=dtype))

        # dr = - a * k_slope * 2π / λ * ((n^2 - 1)/(n^2 + 1))^2
        n2 = n * n
        term = ((n2 - 1.0) / (n2 + 1.0)) ** 2
        dr = -a * k_slope * (2.0 * torch.pi) / wavelengths * term

        spectrum_colored = (spectrum + dr).clamp_(0, 1)
        return spectrum_colored, True
        
def consistency_loss(D, real_samples, material_labels, noise_labels, data_augmentor):
    """Calculates the consistency loss for the discriminator."""
    # Apply data augmentation to real samples
    augmented_samples, _ = data_augmentor.augment(real_samples)
    # Get discriminator scores for augmented samples
    _, d_augmented = D(augmented_samples, material_labels, noise_labels)
    # Get discriminator scores for original real samples
    _, d_real = D(real_samples, material_labels, noise_labels)
    # Stop gradients for the real samples
    d_real = d_real.detach()
    # Consistency loss is the difference between the two scores
    loss = nn.MSELoss()(d_augmented, d_real)
    return loss * cl_coeff  # Scale by the consistency loss coefficient

def main():
    starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # create output dirs
    os.makedirs('img', exist_ok=True)
    os.makedirs('nets', exist_ok=True)

    # load data
    trainset = Dataset_expand("PlasticDataset/small_sets/filtered_all_materials.csv", 
                              print_info=True, expand_factor=False, ignore_noise_labels=True, filter_materials=list(material_labels.keys()))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize data augmentor
    data_augmentor = DataAugmentation()

    # initialize networks
    netD = CondCritic1D(isCR=True).to(device)
    netG = CondGen1D(nz, activation=parser.parse_args().activation).to(device)
    netD.apply(init_weights)
    netG.apply(init_weights)

    # fixed noise for visualisation
    fixed_noise = torch.randn(4 * material_num, nz, device=device)
    fixed_material_labels = torch.tensor(list(material_labels.keys()), device=device)
    fixed_material_labels = fixed_material_labels.repeat(4)
    # fixed_noise_labels = torch.tensor(np.repeat(list(noise_labels.keys()), material_num), device=device)
    fixed_noise_labels = torch.zeros_like(fixed_material_labels)

    # optimizers (Adam for WGAN-GP)
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))

    loss_D_var = []  # to store D loss
    loss_G_var = []  # to store G loss
    for epoch in range(1, epoch_num + 1):
        for i, (real, material_label, noise_label) in enumerate(trainloader, start=1):
            real = real.to(device)
            material_label = material_label.to(device)
            noise_label = noise_label.to(device)

            # ---------------------
            #  1) Train Discriminator
            # ---------------------
            for _ in range(n_critic):
                netD.zero_grad()
                # sample noise and generate fake
                noise = torch.randn(batch_size, nz, device=device)
                fake = netG(noise, material_label, noise_label).detach()  # detach to avoid G gradients

                # real scores and fake scores
                d_real, _ = netD(real, material_label, noise_label)
                d_fake, _ = netD(fake, material_label, noise_label)
                d_real = d_real.mean()
                d_fake = d_fake.mean()

                # gradient penalty
                gp = compute_gradient_penalty(netD, real, fake, material_label, noise_label)

                # consistency loss
                cl = consistency_loss(netD, real, material_label, noise_label, data_augmentor)

                # WGAN-GP loss for D
                loss_D = d_fake - d_real + gp + cl
                loss_D.backward()
                optimizerD.step()

                if torch.isnan(loss_D).any() or torch.isinf(loss_D).any():
                    print("Discriminator loss is NaN or Inf, stopping training.")
                    return

            # -----------------
            #  2) Train Generator
            # -----------------
            netG.zero_grad()
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise, material_label, noise_label)
            # G tries to make D think outputs are real => minimize −E[D(fake)]
            D_fake, _ = netD(fake, material_label, noise_label)
            loss_G = -D_fake.mean()
            loss_G.backward()
            optimizerG.step()

            # logging

            print(f"[Epoch {epoch}/{epoch_num}] [Step {i}/{len(trainloader)}] "
                      f"Loss_D: {loss_D.item():.4f}  "
                      f"Loss_G: {loss_G.item():.4f}")

        # save losses
        loss_D_var.append(loss_D.item())
        loss_G_var.append(loss_G.item())    

        # save intermediate plots
        if epoch % 2 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_material_labels, fixed_noise_labels).cpu()
                intermediate_plot(fake, fixed_material_labels, imgname.format(epoch=epoch), material_num=material_num)

        # if epoch > 100 and loss_G.item() < 1:
        #     print("Early stopping triggered")
        #     break

    # save trained networks
    torch.save(netD.state_dict(), net_D_path)
    torch.save(netG.state_dict(), net_G_path)

    # save loss history
    np.save(loss_path + '_D_small.npy', loss_D_var)
    np.save(loss_path + '_G_small.npy', loss_G_var)

    plt.figure()
    plt.plot(loss_D_var, label='D Loss')
    plt.plot(loss_G_var, label='G Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ylim(-10, 10)
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('loss/loss_curves_small.png')
    plt.close()

    gifkeyword='wgan_gp'
    gifname='smallset.gif'
    os.makedirs('gif', exist_ok=True)
    with imageio.get_writer(os.path.join('gif', gifname), mode='I', duration=1) as writer:
        imgfolder_path = os.path.join('img')
        for gifname in sorted(os.listdir(imgfolder_path)):
            if gifname.endswith('.png') and gifkeyword in gifname:
                image_path = os.path.join(imgfolder_path, gifname)
                image = imageio.imread(image_path)
                writer.append_data(image)
                os.remove(image_path)  # Remove the image after adding to GIF


def visualize_results(seed_real=42, seed_fake=37, noise_in = None, col_num=5):
    trainset = Dataset_expand("PlasticDataset/small_sets/filtered_all_materials.csv", 
                            print_info=True, expand_factor=False, filter_materials=list(material_labels.keys()))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_fake = batch_size
    netG = CondGen1D(nz, activation=parser.parse_args().activation).to(device)
    netG.load_state_dict(torch.load(net_G_path))
    netG.eval()
    if noise_in is not None:
        noise = noise_in
    else:
        noise = torch.randn(num_fake, nz, generator=torch.Generator().manual_seed(seed_fake)).to(device)
    mat_labels = torch.ones(num_fake, device=device, dtype=torch.int) * material_labels.keys().__iter__().__next__()
    ns_labels = torch.zeros(num_fake, device=device, dtype=torch.int)
    fake_plt = netG(noise, mat_labels, ns_labels).cpu().detach().numpy()

    loader_iter = iter(trainloader)
    real_plt = next(loader_iter)[0].cpu()
    data_augmentor = DataAugmentation()
    augmented_real, _ = data_augmentor.augment(real_plt)

    real_plt = real_plt.detach().numpy()

    f, axs = plt.subplots(4, col_num, figsize=(10, 6))
    axs = axs.flatten()
    for i in range(col_num*4):
        if i < col_num*2:
            axs[i].plot(real_plt[i].flatten(), color='blue')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_ylim(0, 1)
        else:
            axs[i].plot(fake_plt[i - col_num*2].flatten(), color='red')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_ylim(0, 1)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'img/real_vs_fake_{material_labels[parser.parse_args().selected_material]}.png')
    plt.close()

    # save_noise(noise.cpu(), [0,2,6,7], [2,2,2,2])

def save_noise(noise, indices, mat_labels, save_dir="temporary", prefix="noise"):

    assert len(indices) == len(mat_labels), "indices and labels must match in length"
    os.makedirs(save_dir, exist_ok=True)
    
    save_dict = {}
    for idx, label in zip(indices, mat_labels):
        if label not in save_dict:
            save_dict[label] = []
        save_dict[label].append(noise[idx])

    # find all existing files that match prefix
    existing_files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pt")]

    # extract numbers using regex
    numbers = []
    for f in existing_files:
        match = re.search(r"_(\d+)\.pt$", f)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers) + 1 if numbers else 1

    # final file name
    file_name = f"{prefix}_{next_num}.pt"
    path = os.path.join(save_dir, file_name)
    
    torch.save(save_dict, path)
    print(f"Saved {len(save_dict)} items to {path}")
    return path

def load_noise(read_dir="temporary"):
    merged_dict = {}

    for f in os.listdir(read_dir):
        data = torch.load(os.path.join(read_dir, f))
        for key, _ in data.items():
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key] += data[key]
            
    # combine into a tensor
    for key in merged_dict:
        combined_noise = torch.stack(merged_dict[key], dim=0)

    return combined_noise

if __name__ == '__main__':
    main()
    # 37 [0,1,4,7]
    # noise_in = load_noise()
    # visualize_results(seed_real=42, seed_fake=43, noise_in=noise_in.to(device))

    visualize_results(seed_real=42, seed_fake=45)

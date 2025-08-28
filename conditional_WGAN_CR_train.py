import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from loaddataset import Dataset_expand
from material_dict import material_labels

from conditional_WGAN_net import init_weights
from conditional_WGAN_net import CondCritic1D_2labels as CondCritic1D
from conditional_WGAN_net import CondGen1D_Upsample_2labels as CondGen1D
from material_dict import material_labels, color_labels, noise_labels

parser = argparse.ArgumentParser(description='Hyperparameters for Conditional WGAN-GP')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
parser.add_argument('--lr_D', type=float, default=5e-5, help='learning rate for Adam optimizer (Discriminator)')
parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for Adam optimizer (Generator)')
parser.add_argument('--gp_coeff', type=float, default=10, help='gradient penalty coefficient')
parser.add_argument('--cl_coeff', type=float, default=10, help='consistency loss coefficient')
parser.add_argument('--n_critic', type=int, default=3, help='D updates per G update')
parser.add_argument('--nz', type=int, default=100, help='noise dimension')
parser.add_argument('--epoch_num', type=int, default=2, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
parser.add_argument('--model_info', type=str, default='_k4_CR', help='information about the model')
parser.add_argument('--islog', type=bool, default=False, help='whether to log the training process')
parser.add_argument('--trainset', type=str, choices=['large', 'small'], default='large', help='which trainset to use')
parser.add_argument('--ignore_noise_labels', type=bool, default=False, help='whether to ignore noise labels')

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
net_G_path = f'nets/netG_con_wgan{suffix}.pth'
net_D_path = f'nets/netD_con_wgan{suffix}.pth'
log_path = f'log/spectral_fid_values{suffix}.md'

if parser.parse_args().trainset == 'small':
    trainset_path = 'PlasticDataset/small_sets/filtered_all_materials.csv'
else:
    trainset_path = 'PlasticDataset/labeled_10materials/merged.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

material_num = 11

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
    f, axes = plt.subplots(4, material_num, figsize=(14, 8), sharex=True, sharey=True)
    for rowidx in range(4):
        for colidx in range(material_num):
            idx = rowidx * material_num + colidx
            axes[rowidx][colidx].plot(fake[idx].flatten().numpy())
            axes[rowidx][colidx].set_xticks([])
            axes[rowidx][colidx].set_yticks([])
            if idx < material_num:
                axes[rowidx][colidx].set_title(material_labels[fixed_labels[idx].item()])
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
                noisetype = random.choice([1, 2, 3])
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

    def addcolor(self, spectrum, color='random', relative_concentration=(0.1, 0.6)):
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
    if cl_coeff == 0:
        return torch.tensor(0.0, device=device)
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
    trainset = Dataset_expand(trainset_path, print_info=True, expand_factor=True, ignore_noise_labels=parser.parse_args().ignore_noise_labels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize data augmentor
    data_augmentor = DataAugmentation()

    # initialize networks
    netD = CondCritic1D(isCR=True, dropout=0.3).to(device)
    netG = CondGen1D(nz).to(device)
    netD.apply(init_weights)
    netG.apply(init_weights)

    # fixed noise for visualisation
    fixed_noise = torch.randn(4 * material_num, nz, device=device)
    fixed_material_labels = torch.tensor(list(material_labels.keys()), device=device)
    fixed_material_labels = fixed_material_labels.repeat(4)
    if parser.parse_args().ignore_noise_labels:
        fixed_noise_labels = torch.zeros(4 * material_num, dtype=torch.int, device=device)
    else:
        fixed_noise_labels = torch.tensor(np.repeat(list(noise_labels.keys()), material_num), device=device)

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
            if i % 5 == 0:
                
                if gp.item() != 0:
                    cr_gp_ratio = f"{(cl.item())/(gp.item()):.4f}"
                else:
                    cr_gp_ratio = "N/A"
                
                print(f"[Epoch {epoch}/{epoch_num}] [Step {i}/{len(trainloader)}] "
                      f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f} "
                      f"CR/GP: {cr_gp_ratio}")
                
            # save losses
            loss_D_var.append(loss_D.item())
            loss_G_var.append(loss_G.item())    

        # save intermediate plots
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_material_labels, fixed_noise_labels).cpu()
            intermediate_plot(fake, fixed_material_labels, imgname.format(epoch=epoch))

    # save trained networks
    torch.save(netD.state_dict(), net_D_path)
    torch.save(netG.state_dict(), net_G_path)
    # save loss history
    np.save('loss/wgan_gp_loss_D.npy', loss_D_var)
    np.save('loss/wgan_gp_loss_G.npy', loss_G_var)

    if not parser.parse_args().islog:
        return

    hyperparams = {
        'beta1': beta1,
        'beta2': beta2,
        'lr_D': lr_D,
        'lr_G': lr_G,
        'gp_coeff': gp_coeff,
        'cl_coeff': cl_coeff,
        'n_critic': n_critic,
        'nz': nz,
        'epoch_num': epoch_num,
        'batch_size': batch_size
    }
    runningtime = datetime.datetime.now() - datetime.datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    newline = "\n"

    if os.path.exists(log_path):
        with open(log_path, 'a') as md:
            md.write(newline * 2)

    with open(log_path, 'a') as md:
        md.write('# Training Log for GAN with Gradient Penalty and Consistency Regularization' + newline + newline)
        md.write('> ### TimeStamp' + newline)
        md.write('> Training started at: ' + starttime + '  ' + newline)
        md.write('> Training finished at: ' + str(datetime.datetime.now()) + '  ' + newline)
        md.write('> Total training time: ' + str(runningtime) + '  ' + newline)
        md.write(newline * 2)

        md.write('> ### Hyperparameters' + newline)
        md.write('> | Parameter | Value |' + newline)
        md.write('> |-----------|-------|' + newline)
        for k, v in hyperparams.items():
            md.write(f'> | {k} | {v} |' + newline)
        md.write(newline * 2)

if __name__ == '__main__':
    main()
    
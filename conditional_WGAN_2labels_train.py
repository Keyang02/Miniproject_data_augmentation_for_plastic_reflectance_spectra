import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

import torch
import torch.optim as optim
import torch.autograd as autograd

from loaddataset import Dataset_expand
from material_dict import material_labels

from conditional_WGAN_net import init_weights
from conditional_WGAN_net import CondCritic1D_2labels as CondCritic1D
from conditional_WGAN_net import CondGen1D_ConvT_2labels as CondGen1D
from material_dict import material_labels, noise_labels

parser = argparse.ArgumentParser(description='Hyperparameters for Conditional WGAN-GP')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for Adam optimizer (Discriminator)')
parser.add_argument('--lr_G', type=float, default=1e-4, help='learning rate for Adam optimizer (Generator)')
parser.add_argument('--p_coeff', type=float, default=10, help='gradient penalty coefficient')
parser.add_argument('--n_critic', type=int, default=3, help='D updates per G update')
parser.add_argument('--nz', type=int, default=100, help='noise dimension')
parser.add_argument('--epoch_num', type=int, default=2, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
parser.add_argument('--model_info', type=str, default='_k4_GP', help='information about the model')
parser.add_argument('--islog', type=bool, default=False, help='whether to log the training process')

# hyperparameters
beta1 = parser.parse_args().beta1               # beta1 for Adam optimizer
beta2 = parser.parse_args().beta2               # beta2 for Adam optimizer
lr_D = parser.parse_args().lr_D                 # learning rate for Adam optimizer (Discriminator)
lr_G = parser.parse_args().lr_G                 # learning rate for Adam optimizer (Generator)
p_coeff = parser.parse_args().p_coeff           # gradient penalty coefficient
n_critic = parser.parse_args().n_critic         # D updates per G update
nz = parser.parse_args().nz                     # noise dimension
epoch_num = parser.parse_args().epoch_num       # number of epochs to train
batch_size = parser.parse_args().batch_size     # batch size for training

suffix = parser.parse_args().model_info         # model information suffix for output files
imgname = 'wgan_gp_epoch_{epoch}.png'
net_G_path = f'nets/netG_con_wgan{suffix}.pth'
net_D_path = f'nets/netD_con_wgan{suffix}.pth'
log_path = f'log/spectral_fid_values{suffix}.md'

material_num = 11  # number of materials

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_gradient_penalty(D, real_samples, fake_samples, material_label, noise_label, oneside = False):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)
    # sample interpolation coefficients in [0,1]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    # create interpolations
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, material_label, noise_label)
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

    penalty = p_coeff * (grad_deviation ** 2).mean()
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

def main():
    starttime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # create output dirs
    os.makedirs('img', exist_ok=True)
    os.makedirs('nets', exist_ok=True)

    # load data
    trainset = Dataset_expand("PlasticDataset/labeled_10materials/merged.csv", print_info=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # initialize networks
    netD = CondCritic1D().to(device)
    netG = CondGen1D(nz).to(device)
    netD.apply(init_weights)
    netG.apply(init_weights)

    # fixed noise for visualisation
    fixed_noise = torch.randn(4 * material_num, nz, device=device)
    fixed_material_labels = torch.tensor(list(material_labels.keys()), device=device)
    fixed_material_labels = fixed_material_labels.repeat(4)
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
                d_real = netD(real, material_label, noise_label).mean()
                d_fake = netD(fake, material_label, noise_label).mean()

                # gradient penalty
                gp = compute_gradient_penalty(netD, real, fake, material_label, noise_label)

                # WGAN-GP loss for D
                loss_D = d_fake - d_real + gp
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
            loss_G = -netD(fake, material_label, noise_label).mean()
            loss_G.backward()
            optimizerG.step()

            # logging
            if i % 5 == 0:
                print(f"[Epoch {epoch}/{epoch_num}] [Step {i}/{len(trainloader)}] "
                      f"Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}")
                
            # save losses
            loss_D_var.append(loss_D.item())
            loss_G_var.append(loss_G.item())    

        # save intermediate plots
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_material_labels, fixed_noise_labels).cpu()
            intermediate_plot(fake, fixed_material_labels, imgname.format(epoch=epoch), material_num=material_num)

    # save trained networks
    torch.save(netD.state_dict(), net_D_path)
    torch.save(netG.state_dict(), net_G_path)
    # save loss history
    np.save('loss/wgan_gp_loss_D.npy', loss_D_var)
    np.save('loss/wgan_gp_loss_G.npy', loss_G_var)

    # log training details
    if not parser.parse_args().islog:
        return
    
    hyperparams = {
        'beta1': beta1,
        'beta2': beta2,
        'lr_D': lr_D,
        'lr_G': lr_G,
        'gp_coeff': p_coeff,
        'n_critic': n_critic,
        'nz': nz,
        'epoch_num': epoch_num,
        'batch_size': batch_size
    }
    
    runningtime = datetime.datetime.now() - datetime.datetime.strptime(starttime, "%Y-%m-%d %H:%M:%S")
    newline = f'{os.linesep}'

    if os.path.exists(log_path):
        with open(log_path, 'a') as md:
            md.write(newline * 2)

    with open(log_path, 'a') as md:
        md.write('# Training Log for GAN with Gradient Penalty' + newline + newline)
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


    
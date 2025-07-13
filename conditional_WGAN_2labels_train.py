import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from loaddataset import Dataset_addnoise
from visualisation import plot_realvsfake, gifplot, plot_loss
from material_dict import material_labels

from conditional_WGAN_net import init_weights
from conditional_WGAN_net import CondCritic1D_2labels as CondCritic1D
from conditional_WGAN_net import CondGen1D_ConvT_2labels as CondGen1D

# hyperparameters
beta1 = 0.0        # beta1 for Adam optimizer
beta2 = 0.9        # beta2 for Adam optimizer
lr = 1e-4           # learning rate for Adam optimizer
p_coeff = 10        # gradient penalty coefficient
n_critic = 3        # D updates per G update
nz = 100            # noise dim

epoch_num = 100
batch_size = 30

imgname = 'wgan_gp_epoch_{epoch}.png'

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

def intermediate_plot(fake, fixed_labels, imgname):
    f, axes = plt.subplots(4, 7, figsize=(14, 8), sharex=True, sharey=True)
    for rowidx in range(4):
        for colidx in range(7):
            idx = rowidx * 7 + colidx
            axes[rowidx][colidx].plot(fake[idx].flatten().numpy())
            axes[rowidx][colidx].set_xticks([])
            axes[rowidx][colidx].set_yticks([])
            if idx < 7:
                axes[rowidx][colidx].set_title(material_labels[fixed_labels[idx].item()])
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(os.path.join('img', imgname))
    plt.close(f)

def main():
    # create output dirs
    os.makedirs('img', exist_ok=True)
    os.makedirs('nets', exist_ok=True)

    # load data
    trainset = Dataset_addnoise("PlasticDataset/labeled/merged_dataset_withlabel.csv", print_info=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # initialize networks
    netD = CondCritic1D().to(device)
    netG = CondGen1D(nz).to(device)
    netD.apply(init_weights)
    netG.apply(init_weights)

    # fixed noise for visualisation
    fixed_noise = torch.randn(28, nz, device=device)
    fixed_material_labels = torch.tensor(np.array([1, 2, 3, 4, 5, 6, 0]), device=device)
    fixed_material_labels = fixed_material_labels.repeat(4)  # repeat to match batch size of 28
    fixed_noise_labels = torch.tensor(np.repeat([1, 2, 3, 4], 7), device=device)

    # optimizers (Adam for WGAN-GP)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

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
            intermediate_plot(fake, fixed_material_labels, imgname.format(epoch=epoch))

    # save trained networks
    torch.save(netD.state_dict(), f'nets/netD_epoch_{epoch_num}.pth')
    torch.save(netG.state_dict(), f'nets/netG_epoch_{epoch_num}.pth')
    # save loss history
    np.save('loss/wgan_gp_loss_D.npy', loss_D_var)
    np.save('loss/wgan_gp_loss_G.npy', loss_G_var)
    # visualisation
    gifplot('wgan_gp.gif', keyword='wgan_gp', isdelete=False)
    print('GIF saved as wgan_gp.gif')
    plot_loss(loss_D_var, loss_G_var)
    print('Loss plot saved as wgan_loss_plot.png')


if __name__ == '__main__':
    main()
    
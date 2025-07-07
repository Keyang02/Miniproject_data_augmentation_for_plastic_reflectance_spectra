import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from wgan import Discriminator, Generator, weights_init
from preprocessing import Dataset_csv
from visualisation import plot_realvsfake, gifplot, plot_loss

# hyperparameters
beta1 = 0.0
beta2 = 0.9
p_coeff = 10        # gradient penalty coefficient
n_critic = 5         # D updates per G update
lr = 1e-4
epoch_num = 500
batch_size = 20
nz = 100       # noise dim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)
    # sample interpolation coefficients in [0,1]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    # create interpolations
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    # ones for grad_outputs
    ones = torch.ones(d_interpolates.size(), device=device)
    # compute gradients wrt interpolates
    grads = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    penalty = p_coeff * ((grad_norm - 1) ** 2).mean()
    return penalty

def main():
    # create output dirs
    os.makedirs('img', exist_ok=True)
    os.makedirs('nets', exist_ok=True)

    # load data
    trainset = Dataset_csv("PlasticDataset/nolabel/merged_dataset.csv")
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # initialize networks
    netD = Discriminator().to(device)
    netG = Generator(nz).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)

    # fixed noise for visualisation
    fixed_noise = torch.randn(16, nz, 1, device=device)

    # optimizers (Adam for WGAN-GP)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    loss_D_var = []  # to store D loss
    loss_G_var = []  # to store G loss
    for epoch in range(1, epoch_num + 1):
        for i, (real, _) in enumerate(trainloader, start=1):
            real = real.to(device)

            # ---------------------
            #  1) Train Discriminator
            # ---------------------
            for _ in range(n_critic):
                netD.zero_grad()
                # sample noise and generate fake
                noise = torch.randn(batch_size, nz, 1, device=device)
                fake = netG(noise).detach()  # detach to avoid G gradients

                # real scores and fake scores
                d_real = netD(real).mean()
                d_fake = netD(fake).mean()

                # gradient penalty
                gp = compute_gradient_penalty(netD, real, fake)

                # WGAN-GP loss for D
                loss_D = d_fake - d_real + gp
                loss_D.backward()
                optimizerD.step()

            # -----------------
            #  2) Train Generator
            # -----------------
            netG.zero_grad()
            noise = torch.randn(batch_size, nz, 1, device=device)
            fake = netG(noise)
            # G tries to make D think outputs are real => minimize −E[D(fake)]
            loss_G = -netD(fake).mean()
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
            fake = netG(fixed_noise).cpu()
            fig, axes = plt.subplots(4, 4, figsize=(8, 8), sharex=True, sharey=True)
            for idx, ax in enumerate(axes.flatten()):
                ax.plot(fake[idx].view(-1).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(f'img/wgan_gp_epoch_{epoch}.png')
            plt.close(fig)


    # save loss history
    np.save('loss/wgan_gp_loss_D.npy', np.array(loss_D_var))
    np.save('loss/wgan_gp_loss_G.npy', np.array(loss_G_var))

    # save final models
    torch.save(netG.state_dict(), 'nets/wgan_gp_netG.pth')
    torch.save(netD.state_dict(), 'nets/wgan_gp_netD.pth')
    print("Training complete. Models saved in ./nets/")

    # plot real vs fake signals
    plot_realvsfake(trainloader, netG, device, nz=nz)

    # plot loss curves
    plot_loss(np.array(loss_D_var), np.array(loss_G_var))

    # create GIF of training process
    gifplot('WGAN-GP.gif')


if __name__ == "__main__":
    main()
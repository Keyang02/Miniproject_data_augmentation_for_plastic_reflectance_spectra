import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from wgan import Discriminator, Generator, weights_init
from preprocessing import Dataset_csv
from visualisation import plot_realvsfake, gifplot, plot_loss

n_critic = 5
clip_value = 0.01
lr = 1e-4
epoch_num = 300
batch_size = 20
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The operating device is " + str(device))

def main():
    # load training data
    trainset = Dataset_csv("PlasticDataset/merged_dataset.csv")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    
    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)

    # optimizers
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    # loss over epochs
    loss_D_var = []
    loss_G_var = []

    # training
    istrain = True
    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(trainloader):
            # training netD
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)

            loss_D = -torch.mean(netD(real_cpu)) + torch.mean(netD(fake))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if step % n_critic == 0:
                # training netG
                noise = torch.randn(b_size, nz, 1, device=device)

                netG.zero_grad()

                fake = netG(noise)

                loss_G = -torch.mean(netD(fake))
                loss_G.backward()
                optimizerG.step()

                loss_D_var.append(loss_D.item())
                loss_G_var.append(loss_G.item())

                if loss_G.item() < -0.4 and loss_D.item() < -1.5 and epoch > 40:
                       # istrain = False
                    break
            
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                % (epoch, epoch_num, step, len(trainloader), loss_D.item(), loss_G.item()),
                end = '\r', flush=True) 

        # save training process
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            f, a = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
                    plt.ylim(0, 1)
            plt.savefig('./img/wgan_epoch_%d.png' % epoch)
            plt.close()
        
        if not istrain:
            print(f'\nTraining stopped at epoch {epoch}.')
            break



    # save model
    torch.save(netG, './nets/wgan_netG.pkl')
    torch.save(netD, './nets/wgan_netD.pkl')

    # save loss
    np.save('./loss/wgan_loss_D.npy', np.array(loss_D_var))
    np.save('./loss/wgan_loss_G.npy', np.array(loss_G_var))

    # visualize trained results
    gifplot('WGAN-GP.gif')
    plot_loss(np.array(loss_D_var), np.array(loss_G_var))
    plot_realvsfake(trainloader, netG, device, nz)


if __name__ == '__main__':
    main()

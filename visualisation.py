import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import torch
from wgan import Generator
from preprocessing import Dataset_csv

def gifplot(root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    with imageio.get_writer(os.path.join(root, 'gif/WGAN.gif'), mode='I', duration=1) as writer:
        imgfolder_path = os.path.join(root, 'img')
        for filename in sorted(os.listdir(imgfolder_path)):
            if filename.endswith('.png') and 'wgan' in filename:
                image_path = os.path.join(imgfolder_path, filename)
                image = imageio.imread(image_path)
                writer.append_data(image)
                os.remove(image_path)  # Remove the image after adding to GIF
    

def plot_loss(root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    loss_D = np.load(os.path.join(root, 'loss/wgan_loss_D.npy'))
    loss_G = np.load(os.path.join(root, 'loss/wgan_loss_G.npy'))

    plt.figure(figsize=(10, 5))
    plt.plot(loss_D, label='Loss D')
    plt.plot(loss_G, label='Loss G')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('WGAN Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(root, 'loss/wgan_loss_plot.png'))
    plt.close()

def plot_realvsfake(trainloader, netG, device, nz=100, root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    for data, _ in trainloader:
        real_data = data.detach().numpy()
        batch_size = real_data.shape[0]
        noise = torch.randn(batch_size, nz, 1, device=device)
        fake_data = netG(noise).detach().numpy()
        break

    f, ax = plt.subplots(4, 4, figsize=(10, 6))
    for i in range(4):
        for j in range(4):
            if i * 4 + j < batch_size:
                ax[i][j].plot(real_data[i * 4 + j].flatten(), label='Real', color='blue')
                ax[i][j].legend('real')
            else:
                ax[i][j].plot(fake_data[i * 4 + j - 8].flatten(), label='Fake', color='red')
                ax[i][j].legend('fake')
            ax[i][j].set_xticks(())
            ax[i][j].set_yticks((0,1))
    plt.suptitle('Real vs Fake Signals')
    plt.savefig(os.path.join(root, 'img/real_vs_fake.png'))
    plt.close()
            
    

if __name__ == '__main__':
    root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'
    gifplot(root = root)
    print('GIF saved as WGAN.gif')
    plot_loss(root = root)
    print('Loss plot saved as wgan_loss_plot.png')

    trainset = Dataset_csv("PlasticDataset/interpolated_spectra_clean.csv")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=8, shuffle=True
        )

    torch.serialization.add_safe_globals([Generator])
    netG = torch.load(os.path.join(root, 'nets/wgan_netG.pkl'), weights_only = False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plot_realvsfake(trainloader, netG, device=device, root = root)
    print('Real vs Fake plot saved as real_vs_fake.png')

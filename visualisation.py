import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import torch
from wgan import Generator
from preprocessing import Dataset_csv
from material_dict import material_labels
from conditional_WGAN_net import CondGen1D, CondGen1D_ConvT
from loaddataset import Dataset_csv

def gifplot(filename, keyword = 'wgan', root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal',isdelete=True):
    os.makedirs('gif', exist_ok=True)
    with imageio.get_writer(os.path.join(root, 'gif', filename), mode='I', duration=1) as writer:
        imgfolder_path = os.path.join(root, 'img')
        for filename in sorted(os.listdir(imgfolder_path)):
            if filename.endswith('.png') and keyword in filename:
                image_path = os.path.join(imgfolder_path, filename)
                image = imageio.imread(image_path)
                writer.append_data(image)
                if isdelete:
                    os.remove(image_path)  # Remove the image after adding to GIF

def plot_loss(loss_D, loss_G, root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal', ylim = -1):

    plt.figure(figsize=(10, 5))
    plt.plot(loss_D, label='Loss D')
    plt.plot(loss_G, label='Loss G')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('WGAN Loss')
    if ylim != -1:
        plt.ylim(ylim)
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
            if i * 4 + j < 8:
                ax[i][j].plot(real_data[i * 4 + j].flatten(), label='Real', color='blue')
                ax[i][j].legend('real')
            else:
                ax[i][j].plot(fake_data[i * 4 + j - 8].flatten(), label='Fake', color='red')
                ax[i][j].legend('fake')
            ax[i][j].set_xticks(())
            ax[i][j].set_yticks((0,1))
    plt.suptitle('Real vs Fake Signals')
    plt.savefig(os.path.join(root, 'img/real_vs_fake.png'))
    plt.close(f)

def sample_one_label(device, label, batchsize, nz=100,
                     dataset_path='PlasticDataset/labeled/merged_dataset_withlabel.csv'):
    real_dataset = Dataset_csv(dataset_path)
    dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(dataloader)

    real = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    while len(real) < batchsize:
        try:
            data_try, label_try = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
        if label_try.item() == label:
            real = torch.cat((real, data_try.to(device)), dim=0)
            labels = torch.cat((labels, label_try.to(device)), dim=0)
        labels = labels.to(torch.int)
    return real, labels

def plot_realvsfake_cond(netG, device, label, batchsize=8, nz=100, 
                         root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):

    real, labels = sample_one_label(device, label, batchsize=batchsize, nz=nz)
    real = real.detach().numpy()

    noise = torch.randn(batchsize, nz, device=device)
    fake = netG(noise, labels).detach().numpy()

    f, ax = plt.subplots(4, 4, figsize=(10, 6))
    f.suptitle(f'Real vs Fake Signals for {material_labels[label]} Reflectance Spectra')
    for i in range(4):
        for j in range(4):
            if i * 4 + j < 8:
                ax[i][j].plot(real[i * 4 + j].flatten(), label='Real', color='blue')
                ax[i][j].legend('real')
                
            else:
                ax[i][j].plot(fake[i * 4 + j - 8].flatten(), label='Fake', color='red')
                ax[i][j].legend('fake')
            ax[i][j].set_xticks(())
            ax[i][j].set_yticks((0,1))
    plt.savefig(os.path.join(root, f'img/real_vs_fake_{material_labels[label]}.png'))
    plt.close(f)

def plot_realvsfake_cond_all(netG, device, batchsize=2, nz=100,
                             root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    label_list = torch.tensor([1, 2, 3, 4, 5, 6, 0], dtype=torch.int).to(device)
    real = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    for label in label_list:
        r, l = sample_one_label(device, label, batchsize=batchsize, nz=nz)
        real = torch.cat((real, r), dim=0)
        labels = torch.cat((labels, l), dim=0)
    real = real.detach().numpy()
    labels = labels.to(torch.int)

    noise = torch.randn(batchsize * len(label_list), nz, device=device)
    fake = netG(noise, labels).detach().numpy()

    f, ax = plt.subplots(4, 7, figsize=(14, 8), sharex=True, sharey=True)
    for colidx in range(7):
        for rowidx in range(4):
            idx = rowidx * 7 + colidx
            if idx < 14:
                ax[rowidx][colidx].plot(real[idx].flatten(), label='Real', color='blue')
                ax[rowidx][colidx].legend('real')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
                if idx < 7:
                    ax[rowidx][colidx].set_title(material_labels[label_list[colidx].item()])
            else:
                ax[rowidx][colidx].plot(fake[idx - 14].flatten(), label='Fake', color='red')
                ax[rowidx][colidx].legend('fake')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
    plt.savefig(os.path.join(root, 'img/real_vs_fake_all_materials.png'))
    plt.close(f)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'

    gifplot('wgan_gp.gif', root = root)
    print('GIF saved as wgan_gp.gif')

    loss_D = np.load(os.path.join(root, 'loss/wgan_gp_loss_D.npy'))
    loss_G = np.load(os.path.join(root, 'loss/wgan_gp_loss_G.npy'))
    plot_loss(loss_D, loss_G, root = root)
    print('Loss plot saved as wgan_loss_plot.png')

    netG = CondGen1D_ConvT(100).to(device)
    netG.load_state_dict(torch.load(os.path.join(root, 'nets/netG_epoch_64.pth')))

    plot_realvsfake_cond(netG, device, 5, batchsize=8, nz=100, root=root)
    print('Real vs Fake plot saved as real_vs_fake_Material.png')

    plot_realvsfake_cond_all(netG, device, batchsize=2, nz=100, root=root)
    print('Real vs Fake plot for all materials saved as real_vs_fake_all_materials.png')


    # trainset = Dataset_csv("PlasticDataset/nolabel/merged_dataset.csv")

    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=8, shuffle=True
    #     )

    # netG = Generator(100)
    # netG.load_state_dict(torch.load(os.path.join(root, 'nets/wgan_gp_netG.pth')))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # plot_realvsfake(trainloader, netG, device=device, root = root)
    # print('Real vs Fake plot saved as real_vs_fake.png')

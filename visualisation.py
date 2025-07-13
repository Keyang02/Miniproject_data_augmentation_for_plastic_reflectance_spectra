import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import torch

from wgan import Generator
from material_dict import material_labels as material_dict
from conditional_WGAN_net import CondGen1D_ConvT_2labels as CondGen1D
from loaddataset import Dataset_addnoise, Dataset_csv

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
    plt.xlabel('Epochs')
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

def sample_one_label(device, label, batchsize, 
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

    real, labels = sample_one_label(device, label, batchsize=batchsize)
    real = real.detach().numpy()

    noise = torch.randn(batchsize, nz, device=device)
    fake = netG(noise, labels).detach().numpy()

    f, ax = plt.subplots(4, 4, figsize=(10, 6))
    f.suptitle(f'Real vs Fake Signals for {material_dict[label]} Reflectance Spectra')
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
    plt.savefig(os.path.join(root, f'img/real_vs_fake_{material_dict[label]}.png'))
    plt.close(f)

def plot_realvsfake_cond_all(netG, device, batchsize=2, nz=100,
                             label_list = [1, 2, 3, 4, 5, 6, 0],
                             root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    label_length = len(label_list)
    label_list = torch.tensor(label_list, dtype=torch.int).to(device)
    real = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    for label in label_list:
        r, l = sample_one_label(device, label, batchsize=batchsize)
        real = torch.cat((real, r), dim=0)
        labels = torch.cat((labels, l), dim=0)
    real = real.detach().numpy()
    labels = labels.to(torch.int)

    noise = torch.randn(batchsize * label_length, nz, device=device)
    fake = netG(noise, labels).detach().numpy()

    f, ax = plt.subplots(4, label_length, figsize=(14, 8), sharex=True, sharey=True)
    for colidx in range(label_length):
        for rowidx in range(2 * batchsize):
            if rowidx < batchsize:
                idx = rowidx + colidx * batchsize
                ax[rowidx][colidx].plot(real[idx].flatten(), label='Real', color='blue')
                ax[rowidx][colidx].legend('real')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
                if rowidx < 1:
                    ax[rowidx][colidx].set_title(material_dict[label_list[colidx].item()])
            else:
                idx = rowidx + colidx * batchsize - batchsize
                ax[rowidx][colidx].plot(fake[idx].flatten(), label='Fake', color='red')
                ax[rowidx][colidx].legend('fake')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
    plt.savefig(os.path.join(root, 'img/real_vs_fake_all_materials.png'))
    plt.close(f)

def sample_multiple_labels(device, material_label, batchsize, 
                     dataset_path='PlasticDataset/labeled/merged_dataset_withlabel.csv'):
    real_dataset = Dataset_addnoise(dataset_path)
    dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(dataloader)

    real = torch.tensor([]).to(device)
    material_labels = torch.tensor([]).to(device)
    noise_labels = torch.tensor([]).to(device)
    while len(real) < batchsize:
        try:
            data_try, label_try, noise_try = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
        if label_try.item() == material_label:
            real = torch.cat((real, data_try.to(device)), dim=0)
            material_labels = torch.cat((material_labels, label_try.to(device)), dim=0)
            noise_labels = torch.cat((noise_labels, noise_try.to(device)), dim=0)
    material_labels = material_labels.to(torch.int)
    noise_labels = noise_labels.to(torch.int)
    return real, material_labels, noise_labels

def plot_realvsfake_cond_all_2labels(netG, device, batchsize=2, nz=100,
                             label_list = [1, 2, 3, 4, 5, 6, 0],
                             root='/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'):
    label_length = len(label_list)
    label_list = torch.tensor(label_list, dtype=torch.int).to(device)
    real = torch.tensor([]).to(device)
    material_labels = torch.tensor([]).to(device)
    noise_labels = torch.tensor([]).to(device)
    for label in label_list:
        r, m, n = sample_multiple_labels(device, label, batchsize=batchsize)
        real = torch.cat((real, r), dim=0)
        material_labels = torch.cat((material_labels, m), dim=0)
        noise_labels = torch.cat((noise_labels, n), dim=0)
    real = real.detach().numpy()
    material_labels = material_labels.to(torch.int)
    noise_labels = noise_labels.to(torch.int)

    noise = torch.randn(batchsize * label_length, nz, device=device)
    fake = netG(noise, material_labels, noise_labels).detach().numpy()

    f, ax = plt.subplots(4, label_length, figsize=(14, 8), sharex=True, sharey=True)
    for colidx in range(label_length):
        for rowidx in range(2 * batchsize):
            if rowidx < batchsize:
                idx = rowidx + colidx * batchsize
                ax[rowidx][colidx].plot(real[idx].flatten(), label='Real', color='blue')
                ax[rowidx][colidx].legend('real')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
                if rowidx < 1:
                    ax[rowidx][colidx].set_title(material_dict[label_list[colidx].item()])
            else:
                idx = rowidx + colidx * batchsize - batchsize
                ax[rowidx][colidx].plot(fake[idx].flatten(), label='Fake', color='red')
                ax[rowidx][colidx].legend('fake')
                ax[rowidx][colidx].set_xticks(())
                ax[rowidx][colidx].set_yticks((0,1))
    plt.savefig(os.path.join(root, 'img/real_vs_fake_all_materials.png'))
    plt.close(f)


def visualise_condWGAN_1label():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'

    gifplot('wgan_gp.gif', root = root)
    print('GIF saved as wgan_gp.gif')

    loss_D = np.load(os.path.join(root, 'loss/wgan_gp_loss_D.npy'))
    loss_G = np.load(os.path.join(root, 'loss/wgan_gp_loss_G.npy'))
    plot_loss(loss_D, loss_G, root = root)
    print('Loss plot saved as wgan_loss_plot.png')

    netG = CondGen1D(100).to(device)
    netG.load_state_dict(torch.load(os.path.join(root, 'nets/netG_epoch_64.pth')))

    for m in range(7):
        plot_realvsfake_cond(netG, device, m, batchsize=8, nz=100, root=root)
        print(f'Real vs Fake plot saved as real_vs_fake_{material_dict[m]}.png')

    plot_realvsfake_cond_all(netG, device, batchsize=2, nz=100, root=root)
    print('Real vs Fake plot for all materials saved as real_vs_fake_all_materials.png')

def visualise_condWGAN_2labels():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = '/homes/kw635/spectrum_gen_1D/GANs-for-1D-Signal'

    gifplot('wgan_gp.gif', root = root)
    print('GIF saved as wgan_gp.gif')

    loss_D = np.load(os.path.join(root, 'loss/wgan_gp_loss_D.npy'))
    loss_G = np.load(os.path.join(root, 'loss/wgan_gp_loss_G.npy'))
    plot_loss(loss_D, loss_G, root = root)
    print('Loss plot saved as wgan_loss_plot.png')

    netG = CondGen1D(100).to(device)
    netG.load_state_dict(torch.load(os.path.join(root, 'nets/netG_epoch_100.pth')))

    # for m in range(7):
    #     plot_realvsfake_cond(netG, device, m, batchsize=8, nz=100, root=root)
    #     print(f'Real vs Fake plot saved as real_vs_fake_{material_labels[m]}.png')

    plot_realvsfake_cond_all_2labels(netG, device, batchsize=2, nz=100, root=root)
    print('Real vs Fake plot for all materials saved as real_vs_fake_all_materials.png')

if __name__ == '__main__':
    visualise_condWGAN_2labels()
    print('Visualisation completed.')

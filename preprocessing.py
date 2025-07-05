import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

class Dataset_csv(Dataset):
    def __init__(self, root):
        self.dataset = self.build_dataset(root)
        # self.minmax_normalize()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint = self.dataset[idx, :]
        datapoint = torch.unsqueeze(datapoint, 0)
        # target = self.label[idx]
        label = 0  # only one class
        return datapoint, label

    def build_dataset(self, root):
        '''get dataset of signal'''
        df = pd.read_csv(root)
        # df_oneclass = df[df['Material'] == 2.0]
        # df_oneclass = df_oneclass.drop(columns=['Material'])
        dataset = torch.tensor(df.values, dtype=torch.float32)
        return dataset

    def minmax_normalize(self):
        '''return minmax normalize dataset'''
        for idx in range(self.length):
            self.dataset[:, idx] = (self.dataset[:, idx] - self.dataset[:, idx].min()) / (
                self.dataset[:, idx].max() - self.dataset[:, idx].min())

class Dataset_txt():
    def __init__(self, root):
        self.root = root
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = 0  # only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''
        dataset = []
        for _file in os.listdir(self.root):
            sample = np.loadtxt(os.path.join(self.root, _file), comments='#',delimiter=',').T
            dataset.append(sample)
        dataset = np.vstack(dataset).T
        dataset = torch.from_numpy(dataset).float()

        return dataset

    def minmax_normalize(self):
        '''return minmax normalize dataset'''
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    dataset = Dataset_txt('./data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()

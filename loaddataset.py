import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class Dataset_csv(Dataset):
    def __init__(self, root):
        self.dataset, self.labels = self.build_dataset(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint = self.dataset[idx, :]
        datapoint = torch.unsqueeze(datapoint, 0)
        label = self.labels[idx]
        return datapoint, label

    def build_dataset(self, root):
        '''get dataset of signal'''
        df = pd.read_csv(root)
        print(f"The number of each material: {df['Material'].value_counts()}")
        labels = df['Material'].values.copy()
        features = df.drop(columns=['Material']).values
        labels = torch.tensor(labels, dtype=torch.int)
        features = torch.tensor(features, dtype=torch.float32)
        return features, labels

if __name__ == '__main__':
    dataset = Dataset_csv('PlasticDataset/labeled/merged_dataset_withlabel.csv')
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    print(f"Dataset shape: {dataset.dataset.shape}")
    print(f"Labels shape: {dataset.labels.shape}")
    datapoint, label = dataset.__getitem__(0)
    print(f"First datapoint shape: {datapoint.shape}")
    print(f"First label: {label}")


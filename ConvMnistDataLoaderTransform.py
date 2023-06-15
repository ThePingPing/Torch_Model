import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

class PrivateCustomDataSet(Dataset):
    def __init__(self, tensors, transform=None):
        """only if the data size and the label Match"""
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        """if the transformation work , return X with the transform """
        if self.transform:
            x = self.transform(self.tensors[0][index])
        else:
            x = self.tensors[0][index]

        y = self.tensors[1][index]

        return x, y ## (data, label)

    def __len__(self):
        return self.tensors[0].size(0)



def ConvModelLoader():


    def resize_and_normalization(data_mnist):
        label = data_mnist["label"][0:8].values
        data = data_mnist.drop(["label"], axis=1)
        data = data[0:8].values

        data_normal = data / np.max(data)  ## normalization
        data_normal_2D = data_normal.reshape(data_normal.shape[0], 1, 28, 28) ## reshape to 2D

        return data_normal_2D, label

    def torch_data(data_normal_2D, label):

        label_torch = torch.tensor(label).long()
        data_torch = torch.tensor(data_normal_2D).float()

        return data_torch, label_torch

    def custom_loader (data_torch, label_torch):

        img_transform = T.Compose([T.ToPILImage(), T.RandomVerticalFlip(p=0.5), T.ToTensor()])
        X_train_torch = PrivateCustomDataSet((data_torch, label_torch), img_transform)
        """Check the type is Custom"""
        print(type(X_train_torch))
        data_train_loader = DataLoader(X_train_torch, batch_size=8, shuffle=False)

        return data_train_loader

    def take_batche_and_plot(data_train_loader, data_torch):
        X, y = next(iter(data_train_loader))
        fig, axs = plt.subplots(2, 8, figsize=(16, 4))


        """Loop and plot the original , and the data in the loader"""
        print(data_torch.shape)

        for i in range(len(X)):
            axs[0, i].imshow(data_torch[i, 0, :, :].detach(), cmap="gray")
            axs[1, i].imshow(X[i, 0, :, :].detach(), cmap="gray")

            for row in range(2):
                axs[row, i].set_xticks([])
                axs[row, i].set_yticks([])

        axs[0,0].set_ylabel("original")
        axs[1,0].set_ylabel("Loader")

        plt.show()


    data_mnist = pd.read_csv('mnist_train.csv')
    print(data_mnist.head())
    data_normal_2D, label = resize_and_normalization(data_mnist)
    data_torch, label_torch = torch_data(data_normal_2D, label)
    data_train_loader = custom_loader(data_torch, label_torch)
    take_batche_and_plot(data_train_loader, data_torch)






if __name__ == '__main__':
    ConvModelLoader()
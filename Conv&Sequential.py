import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch.optim import Adam

from sklearn.model_selection import train_test_split

from torchsummary import summary


def CreateTheGaussModelConv():

    class GaussNet(nn.Module):

        def __init__(self):
            super().__init__()

            # all layers in one go using nn.Sequential
            self.enc = nn.Sequential(
                nn.Conv2d(1, 6, 3, padding=1),  # output size: (91+2*1-3)/1 + 1 = 91
                nn.ReLU(),  # note that relu is treated like a "layer"
                nn.AvgPool2d(2, 2),  # output size: 91/2 = 45
                nn.Conv2d(6, 4, 3, padding=1),  # output size: (45+2*1-3)/1 + 1 = 45
                nn.ReLU(),  #
                nn.AvgPool2d(2, 2),  # output size: 45/2 = 22
                nn.Flatten(),  # vectorize conv output
                nn.Linear(22 * 22 * 4, 50),  # output size: 50
                nn.Linear(50, 1),  # output size: 1
            )

        def forward(self, x):
            return self.enc(x)

    # create the model instance
    GaussNet = GaussNet()
    return GaussNet


def ConvAndSequential():


    def create_image_data():
        image_per_cat = 1000
        image_size = 91

        x = np.linspace(-4, 4, image_size)
        X,Y = np.meshgrid(x,x)

        width = [1.8, 2.4]

        ## initialization the Tensor

        images = torch.zeros(2*image_per_cat, 1, image_size, image_size)
        labels = torch.zeros(2*image_per_cat)

        """ Implement the Gauss Center"""
        for i in range(2 * image_per_cat):

            ro = 2 * np.random.randn(2)
            G = np.exp(-((X - ro[0]) ** 2 + (Y - ro[1]) ** 2) / (2 * width[i % 2] ** 2))

            """ Put Noise"""
            G = G + np.random.randn(image_size, image_size) / 5

            """ Add to Tensor"""

            images [i, :, :, :] = torch.Tensor(G).view(1, image_size, image_size)
            labels[i] = i%2

        labels = labels[:, None]
        return images, labels, image_per_cat, image_size

    def plot_some_images(images, labels, image_per_cat):

        fig, axs = plt.subplots(3, 7, figsize=(13, 7))

        for i, ax in enumerate(axs.flatten()):
            choice = np.random.randint(2 * image_per_cat)
            G = np.squeeze(images[choice, :, :])
            ax.imshow(G, vmin=-1, vmax=1, cmap="jet")
            ax.set_title('Class %s' %int(labels[choice].item()))
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

    def data_loader_convert(images, labels):

        X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=0.9)

        # then convert them into PyTorch Datasets (note: already converted to tensors)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        batchsize = 32

        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

        print(train_loader.dataset.tensors[0].shape)
        print(train_loader.dataset.tensors[1].shape)

        return train_loader, test_loader

    def initialize_model(learning_rate):
        GaussNet = CreateTheGaussModelConv()
        loss_function = nn.BCEWithLogitsLoss() ## The Data not Normalized so Use BCE With Log
        optimizer = Adam(GaussNet.parameters(), lr=learning_rate)
        return GaussNet, loss_function, optimizer

    def test_model(GaussNet, loss_function, train_loader, image_size):
        X, y = next(iter(train_loader))
        yHat = GaussNet(X)

        print(" ")
        print("Yhat Shape", yHat.shape)
        print("y Shape", y.shape)

        loss = loss_function(yHat, y)
        print(" ")
        print("loss:", loss)
        print(summary(GaussNet,(1, image_size, image_size)))


    def train_model(GaussNet, loss_function, optimizer, train_loader, test_loader):

        # number of epochs
        numepochs = 10

        # initialize losses
        train_Loss = torch.zeros(numepochs)
        test_Loss = torch.zeros(numepochs)
        train_accuracy = torch.zeros(numepochs)
        test_accuracy = torch.zeros(numepochs)

        # loop over epochs
        for epochi in range(numepochs):

            # loop over training data batches
            batch_accuracy = []
            batch_loss = []
            for X, y in train_loader:
                # forward pass and loss
                yHat = GaussNet(X)
                loss = loss_function(yHat, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss from this batch
                batch_loss.append(loss.item())
                batch_accuracy.append(torch.mean(((yHat > 0) == y).float()).item())
            # end of batch loop...

            # and get average losses across the batches
            train_Loss[epochi] = np.mean(batch_loss)
            train_accuracy[epochi] = 100 * np.mean(batch_accuracy)

            # test accuracy
            X, y = next(iter(test_loader))  # extract X,y from test dataloader
            with torch.no_grad():  # deactivates autograd
                yHat = GaussNet(X)
                loss = loss_function(yHat, y)

            # compare the following really long line of code to the training accuracy lines
            test_Loss[epochi] = loss.item()
            test_accuracy[epochi] = 100 * torch.mean(((yHat > 0) == y).float()).item()

        # end epochs

        # function output
        return train_Loss, test_Loss, train_accuracy, test_accuracy


    def plot_train_result(train_Loss, test_Loss, train_accuracy, test_accuracy):

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(train_Loss, 's-', label='Train loss')
        ax[0].plot(test_Loss, 'o-', label='Test loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')
        ax[0].legend()

        ax[1].plot(train_accuracy, 's-', label='Train, accuracy')
        ax[1].plot(test_accuracy, 'o-', label='Test accuracy, ')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Accuracy')
        ax[1].set_title(f'Final model test accuracy: {test_accuracy[-1]:.2f}%')
        ax[1].legend()
        plt.show()

    def plot_test_prediction(GaussNet,test_loader):
        X,y = next(iter(test_loader))
        yHat = GaussNet(X)
        fig, axs = plt.subplots(2, 10, figsize=(13, 4))

        for i, ax in enumerate(axs.flatten()):
            G = torch.squeeze(X[i, 0, :, :]).detach()
            ax.imshow(G, vmin=-1, vmax=1, cmap="jet")
            predict_label = (int(y[i].item()), int(yHat[i].item() > 0))
            ax.set_title('T:%s, P:%s' %predict_label)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    def plot_filter_model(GaussNet):
        filter_conv1 = GaussNet.enc[0].weight
        filter_conv2 = GaussNet.enc[3].weight

        print(filter_conv1.shape) ##--> torch.Size([6, 1, 3, 3])
        print(filter_conv2.shape) ##--> torch.Size([4, 6, 3, 3])
        fig, axs = plt.subplots(1, 6, figsize=(13, 4))

        for i, ax in enumerate(axs.flatten()):
            G = torch.squeeze(filter_conv1[i, 0, :, :]).detach()
            ax.imshow(G, vmin=-1, vmax=1, cmap="Purples")
            ax.axis("off")

        plt.suptitle("The Filters for the first Conv")
        plt.show()

        fig, axs = plt.subplots(4, 6, figsize=(15, 9))

        for i in range(6 * 4):
            index_filter = np.unravel_index(i, (4, 6))
            print(index_filter)
            G = torch.squeeze(filter_conv2[index_filter[0], index_filter[1], :, :]).detach()
            axs[index_filter].imshow(G, vmin=-1, vmax=1, cmap="Purples")
            axs[index_filter].axis('off')

        plt.suptitle('Second convolution layer filters')
        plt.show()


    images, labels, image_per_cat, image_size = create_image_data()
    plot_some_images(images, labels, image_per_cat)
    train_loader, test_loader = data_loader_convert(images, labels)
    learning_rate = 0.001
    GaussNet, loss_function, optimizer = initialize_model(learning_rate)
    test_model(GaussNet, loss_function, train_loader, image_size)
    train_Loss, test_Loss, train_accuracy, test_accuracy = train_model(GaussNet, loss_function, optimizer, train_loader, test_loader)
    plot_train_result(train_Loss, test_Loss, train_accuracy, test_accuracy)
    plot_test_prediction(GaussNet,test_loader)
    plot_filter_model(GaussNet)


if __name__ == '__main__':
    ConvAndSequential()
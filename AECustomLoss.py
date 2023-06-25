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


def CreateCustomLossModel():

    class CnnGaussNetAuto(nn.Module):
        def __init__(self):
            super().__init__()

            ## Encoder Part

            self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),  # output size: (91+2*1-3)/1 + 1 = 91
            nn.ReLU(),  # note that relu is treated like a "layer"
            nn.MaxPool2d(2, 2),  # output size: 91/2 = 45
            nn.Conv2d(6, 4, 3, padding=1),  # output size: (45+2*1-3)/1 + 1 = 45
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

            )

            ## Decoder Part

            self.decoder = nn.Sequential(

                nn.ConvTranspose2d(4, 6, 3, 2),
                nn.ReLU(),
                nn.ConvTranspose2d(6, 1, 3, 2)

            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    CnnAutoModel = CnnGaussNetAuto()

    return CnnAutoModel


def CreateLosses():

    class LossOne(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, yHat, y):
            loss_one = torch.mean(torch.abs(yHat - y))

            return loss_one
    l1 = LossOne()


    class LossTow(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, yHat, y):
            loss_tow = torch.mean((yHat - y)**2) + torch.abs(torch.mean(yHat))

            return loss_tow
    l2 = LossTow()


    class LossThree(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, yHat, y):

            Mu_yHat = torch.mean(yHat)
            Mu_y = torch.mean(y)
            Std_yHat = torch.std(yHat)
            Std_y = torch.std(y)
            numerator = torch.sum((yHat-Mu_yHat) * (y-Mu_y))
            denominator = (torch.numel(y) - 1) * Std_yHat * Std_y

            loss_three = numerator / denominator

            return loss_three

    l3 = LossThree()

    return l1, l2, l3


def CustomLoss():

    def create_image_data():
        image_per_cat = 1000
        image_size = 91

        x = np.linspace(-4, 4, image_size)
        X, Y = np.meshgrid(x, x)

        widths = np.linspace(2, 20, image_per_cat)

        ## initialization the Tensor

        images = torch.zeros(image_per_cat, 1, image_size, image_size)

        """ Implement the Gauss Center"""
        for i in range(image_per_cat):
            ro = 1.5 * np.random.randn(2)
            G = np.exp(-((X - ro[0]) ** 2 + (Y - ro[1]) ** 2) / (widths[i]))

            """ Put Noise"""
            G = G + np.random.randn(image_size, image_size) / 5

            """ add Random horizontal or vertical Bar """

            stat_location_line = np.random.choice(np.arange(2, 28))
            width_line = np.random.choice(np.arange(2, 6))

            boolean_choice = np.random.randn()

            print("stat_location_lign:", stat_location_line, "width",width_line, "boolean_choice ", boolean_choice )

            if boolean_choice > 0:
                G[stat_location_line:stat_location_line + width_line,:] = 1
            else:
                G[:, stat_location_line:stat_location_line + width_line] = 1



            """ Add to Tensor"""

            images[i, :, :, :] = torch.Tensor(G).view(1, image_size, image_size)


        return images, image_per_cat, image_size

    def plot_some_images(images, image_per_cat):

        fig, axs = plt.subplots(3, 7, figsize=(13, 7))

        for i, ax in enumerate(axs.flatten()):
            choice = np.random.randint(image_per_cat)
            G = np.squeeze(images[choice, :, :])
            ax.imshow(G, vmin=-1, vmax=1, cmap="jet")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

    def initialize_model(learning_rate, loss_function_one):
        CnnAutoModel = CreateCustomLossModel()
        loss_function = loss_function_one
        optimizer = Adam(CnnAutoModel.parameters(), lr=learning_rate)
        return CnnAutoModel, loss_function, optimizer



    def test_model(CnnAutoModel, images, image_size):
        yHat = CnnAutoModel(images[:10, :, :, :])

        print(" ")
        print("Yhat Shape", yHat.shape)
        print(" ")
        print(summary(CnnAutoModel,(1, image_size, image_size)))

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].imshow(torch.squeeze(images[0, 0, :, :]).detach(), cmap='jet')
        ax[0].set_title('Model input')
        ax[1].imshow(torch.squeeze(yHat[0, 0, :, :]).detach(), cmap='jet')
        ax[1].set_title('Model output')

        plt.show()

    def train_model(CnnAutoModel, loss_function, optimizer, image_per_cat):

        # number of epochs
        numepochs = 500
        # initialize losses
        losses = torch.zeros(numepochs)

        # loop over epochs
        for epochi in range(numepochs):
            # pick a set of images at random
            batch_image = np.random.choice(image_per_cat, size=32, replace=False)
            X = images[batch_image, :, :, :]

            # forward pass and loss
            yHat = CnnAutoModel(X)
            loss = loss_function(yHat, X)
            losses[epochi] = loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # end epochs

        # function output
        fig, axs = plt.subplots(2, 10, figsize=(18, 4))
        for i in range(10):
            G = torch.squeeze(X[i, 0, :, :]).detach()
            O = torch.squeeze(yHat[i, 0, :, :]).detach()

            axs[0, i].imshow(G, vmin=-1, vmax=1, cmap='jet')
            axs[0, i].axis('off')
            axs[0, i].set_title('Model input')

            axs[1, i].imshow(O, vmin=-1, vmax=1, cmap='jet')
            axs[1, i].axis('off')
            axs[1, i].set_title('Model output')

        plt.show()

        plt.plot(losses, 's-', label='Train')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Model loss (final loss=%.3f)' % losses[-1])

        plt.show()

        return CnnAutoModel, losses


    learning_rate = 0.001
    images, image_per_cat, image_size = create_image_data()
    plot_some_images(images, image_per_cat)
    loss_function_one, loss_function_tow, loss_function_three = CreateLosses()
    CnnAutoModel, loss_function, optimizer = initialize_model(learning_rate, loss_function_three)
    test_model(CnnAutoModel, images, image_size)
    CnnAutoModel, losses = train_model(CnnAutoModel, loss_function, optimizer, image_per_cat)



if __name__ == '__main__':
    CustomLoss()
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torchvision
import torchvision.datasets as datasets


class AutoEncMnist(nn.Module):
    def __init__(self,):
        super().__init__()

        ### input layer
        self.input = nn.Linear(784, 250)

        ### encoder layers
        self.enc = nn.Linear(250, 50)

        ## botelnec layer

        self.lat = nn.Linear(50, 250)

        ### decoder layer
        self.dec = nn.Linear(250, 784)

        # forward pass

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.enc(x))
        x = F.relu(self.lat(x))
        return torch.sigmoid(self.dec(x))

def auto_encoder_mnist_model():

    def scale_and_slip_data(data, scaler_factor):
        mnist_data_scale = data.drop(['label'], axis=1) / scaler_factor
        #mnist_label = data["label"] ## NOT NEED THE LABEL IN AUTOENCODER
        print(mnist_data_scale)

        mnist_data_torch = torch.tensor(mnist_data_scale.values).float()
        return mnist_data_torch

    def initialize_model(optimizer_choice, learning_rate, momentum):
        AutoEncModel = AutoEncMnist()
        loss_function = nn.MSELoss()
        optimizer_function = getattr(torch.optim, optimizer_choice)
        optimizer = optimizer_function(AutoEncModel.parameters(), lr=learning_rate)
        return AutoEncModel, loss_function, optimizer

    def plot_data(data_torch, AutoEncModel):

        X_data = data_torch[:5, :]
        yHat = AutoEncModel(X_data)
        print(X_data.shape, yHat.shape)

        fig, axs = plt.subplots(2, 5, figsize=(10, 3))

        for i in range(5): ## plot multy picture
            axs[0, i].imshow(X_data[i, :].view(28, 28).detach(), cmap='gray')
            axs[1, i].imshow(yHat[i, :].view(28, 28).detach(), cmap='gray')
            axs[0, i].set_xticks([]), axs[0, i].set_yticks([])
            axs[1, i].set_xticks([]), axs[1, i].set_yticks([])

        plt.suptitle('Yikes!!!')
        plt.show()

    def train_model(AutoEncModel, loss_function, optimizer, data_torch ):

        number_epochs = 8000
        losses_list = torch.zeros(number_epochs)

        for epochis in range(number_epochs):
            # select a random set of images
            select_rand_data = np.random.choice(data_torch.shape[0], size=32)
            X = data_torch[select_rand_data, :]

            # forward pass and loss
            yHat = AutoEncModel(X)
            loss = loss_function(yHat, X)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # losses in this epoch
            losses_list[epochis] = loss.item()
            # end epochs

            # function output
        return losses_list, AutoEncModel


    def plot_train_result(losses_list):
        #print(f'Final loss: {losses_list[-1]:.4f}')
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(losses_list)
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')

        plt.show()




    ######## Initialize Parmetre  ###########
    # import dataset (comes with colab!)
    data = pd.read_csv('mnist_train.csv')

    scaler_factor = np.max(data.values)
    print(data.head())
    optimizer_choice = "Adam"
    learning_rate = 0.001
    momentum = 0 ## use Adma Here so don't have momentum
    ###########################

    data_torch = scale_and_slip_data(data, scaler_factor)
    AutoEncModel, loss_function, optimizer = initialize_model(optimizer_choice, learning_rate, momentum)
    plot_data(data_torch, AutoEncModel)

    losses_list, AutoEncModelTrain = train_model(AutoEncModel, loss_function, optimizer, data_torch)
    print("Imhere")
    plot_train_result(losses_list)

    ## Final Show the picture after he trained
    plot_data(data_torch, AutoEncModelTrain)



if __name__ == '__main__':
    auto_encoder_mnist_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

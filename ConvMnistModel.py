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

def CreateTheMnistModelConv(print_toggle=False):

    class MnistNet(nn.Module):

        def __init__(self, print_toggle):
            super().__init__()

            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1) ## 10 out pu chanels , you can change it but tha's have to match with the next layer

            ## the Size output conv1 = np.floor((sizeImg + (2 * padding - Kernel_size) / stride_size ) +1) --> result is 13 for the MaxPOOL

            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)

            ## the Size output conv2 = np.floor((sizeImg + (2 * padding - Kernel_size) / stride_size ) +1) --> result is 5


            """ Comput The number of units in the Fc Layers == number of output from conv2 """

            units_size = np.floor((5+2*0-1)/1) + 1
            units_size = 20 * int(units_size**2)
            print( "The unite Size:",  units_size)

            ### Fully-Connected Layer

            self.fc1 = nn.Linear(units_size, 50)

            ### OutPut layer

            self.output = nn.Linear(50, 10) ## 10 final Features

            self.print = print_toggle

        def forward(self, x):
            print(f'Input : {x.shape}') if self.print else None

            ## Conv --> MaxPool --> Relu

            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            print(f'Layer Conv1 / Pool2-1  : {x.shape}') if self.print else None

            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            print(f'Layer Conv2 / Pool2-2  : {x.shape}') if self.print else None

            ## Compute the number of units

            number_units = x.shape.numel()/ x.shape[0]

            x = x.view(-1, int(number_units))
            if self.print: print(f'To Vector: {x.shape}')

            ## The Linear layer

            x = F.relu(self.fc1(x))
            if self.print: print(f' Layer Fc1: {x.shape}')

            x = self.output(x)
            if self.print: print(f' Layer out: {x.shape}')

            return x

    MnistNet = MnistNet(print_toggle)
    return MnistNet



def ConvMnistModel():

    def resize_and_normalization(data_mnist):
        label = data_mnist["label"].values
        data = data_mnist.drop(["label"], axis=1)
        data = data.values

        data_normal = data / np.max(data)  ## normalization
        data_normal_2D = data_normal.reshape(data_normal.shape[0], 1, 28, 28)  ## reshape to 2D

        print("data shape: " , data_normal_2D.shape, "Label shape :", label.shape)
        return data_normal_2D, label

    def data_loader_convert(data_normal_2D, label):

        label_torch = torch.tensor(label).long()
        data_torch = torch.tensor(data_normal_2D).float()

        X_train, X_test, y_train, y_test = train_test_split(data_torch, label_torch, train_size=0.8)

        # then convert them into PyTorch Datasets (note: already converted to tensors)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        batchsize = 32

        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

        return train_loader, test_loader

    def initialize_model(print_toggle, learning_rate):
        MnistNet = CreateTheMnistModelConv(print_toggle)
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(MnistNet.parameters(), lr=learning_rate)

        return MnistNet, loss_function, optimizer

    def test_model(MnistNet, loss_function, train_loader):
        X, y = next(iter(train_loader))
        yHat = MnistNet(X)

        print(" ")
        print("Yhat Shape", yHat.shape)
        print("y Shape", y.shape)

        loss = loss_function(yHat, y)
        print(" ")
        print("loss:", loss)
        print(summary(MnistNet,(1, 28, 28)))


    def train_model(MnistNet, loss_function, optimizer, train_loader, test_loader):

        number_epochs = 10
        train_accuracy = []
        test_accuracy = []
        losses_list = torch.zeros(number_epochs)

        for epochis in range(number_epochs):
            MnistNet.train()  ## turn on the train
            batch_accuracy = []
            batch_loss = []

            for X, y in train_loader:
                print("Hello im here1")
                yHat = MnistNet(X)
                loss = loss_function(yHat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # take loss from this batch
                batch_loss.append(loss.item())
                matches = torch.argmax(yHat, axis=1) == y  ## prediction vector True/ False
                matches_numerical = matches.float()
                accuracy_score = 100 * torch.mean(matches_numerical)
                batch_accuracy.append(accuracy_score)

            # get the average from train accuracy to each epoch and the loss from the batch
            losses_list[epochis] = np.mean(batch_loss)
            train_accuracy.append(np.mean(batch_accuracy))

            # test accuracy
            MnistNet.eval()  # switch training off
            X, y = next(iter(test_loader))  # extract X,y from test dataloader
            with torch.no_grad():
                yHat = MnistNet(X)
            test_accuracy.append(100 * torch.mean((torch.argmax(yHat, axis=1) == y).float()))
            print("######################################################################Hello im here2")

        print("Done Compute Epochis : --> ", epochis, "From /", number_epochs)

            # function output
        return train_accuracy, test_accuracy, losses_list



    def plot_train_result(train_accuracy, test_accuracy, losses_list):

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(losses_list, 's-')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')

        ax[1].plot(train_accuracy, 's-', label='Train')
        ax[1].plot(test_accuracy, 'o-', label='Test')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Accuracy')
        ax[1].set_title(f'Final model test accuracy: {test_accuracy[-1]:.2f}%')
        ax[1].legend()
        plt.show()


    data_mnist = pd.read_csv('mnist_train.csv')

    ## intialize model Params
    print_toggle = False
    learning_rate = 0.001

    print(data_mnist.head())
    data_normal_2D, label = resize_and_normalization(data_mnist)
    train_loader, test_loader = data_loader_convert(data_normal_2D, label)
    MnistNet, loss_function, optimizer = initialize_model(print_toggle, learning_rate)
    test_model(MnistNet, loss_function, train_loader)
    train_accuracy, test_accuracy, losses_list = train_model(MnistNet, loss_function, optimizer, train_loader, test_loader)
    plot_train_result(train_accuracy, test_accuracy, losses_list)

if __name__ == '__main__':
    ConvMnistModel()

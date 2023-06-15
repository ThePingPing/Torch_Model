import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def createTheQwertyNet(nUnits, nLayers):
    class qwertyNet(nn.Module):
        def __init__(self, nUnits, nLayers):
            super().__init__()

            # create dictionary to store the layers
            self.layers = nn.ModuleDict()
            self.nLayers = nLayers

            ### input layer
            self.layers['input'] = nn.Linear(2, nUnits) ## tow features X, Y coordonate

            ### hidden layers
            for i in range(nLayers):
                self.layers[f'hidden{i}'] = nn.Linear(nUnits, nUnits)

            ### output layer
            self.layers['output'] = nn.Linear(nUnits, 3) ## 3 Category

        # forward pass
        def forward(self, x):
            # input layer
            x = self.layers['input'](x)

            # hidden layers
            for i in range(self.nLayers):
                x = F.relu(self.layers[f'hidden{i}'](x))

            # return output layer
            x = self.layers['output'](x)
            return x

    net = qwertyNet(nUnits, nLayers)
    return net


def data_vs_depth():

    def plot_data(data_torch, label_torch, alpha_factor):
        plt.figure(figsize=(7, 7))
        plt.plot(data_torch[np.where(label_torch == 0)[0], 0], data_torch[np.where(label_torch == 0)[0], 1], 'bs',
                 alpha=alpha_factor)  ## Blue Rectangle
        plt.plot(data_torch[np.where(label_torch == 1)[0], 0], data_torch[np.where(label_torch == 1)[0], 1], 'ko',
                 alpha=alpha_factor)  ## Black Cirecle
        plt.plot(data_torch[np.where(label_torch == 2)[0], 0], data_torch[np.where(label_torch == 2)[0], 1], 'r^',
                 alpha=alpha_factor)  ## Red Rectangle
        plt.title('The qwerties!')
        plt.xlabel('qwerty dimension 1')
        plt.ylabel('qwerty dimension 2')
        plt.show()

    def plot_train_result(train_accuracy, test_accuracy, losses_list):

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(losses_list.detach())
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')

        ax[1].plot(rain_accuracy, label='Train')
        ax[1].plot(test_accuracy, label='Test')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Accuracy')
        ax[1].legend()
        plt.show()

    def train_model(AnnModel, loss_function, optimizer, X_train_loader, X_test_loader):
        number_epochs = 100
        train_accuracy = []
        test_accuracy = []
        losses_list = torch.zeros(number_epochs)

        for epochis in range(number_epochs):
            AnnModel.train() ## turn on the train
            batch_accuracy = []
            batch_loss = []

            for X, y in X_train_loader:

                yHat = AnnModel(X)
                loss = loss_function(yHat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # take loss from this batch
                batch_loss.append(loss.item())
                matches = torch.argmax(yHat, axis=1) == y ## prediction vector True/ False
                matches_numerical = matches.float()
                accuracy_score = 100 * torch.mean(matches_numerical)
                batch_accuracy.append(accuracy_score)



            # get the average from train accuracy to each epoch and the loss from the batch
            losses_list[epochis] = np.mean(batch_loss)
            train_accuracy.append(np.mean(batch_accuracy))

            # test accuracy
            AnnModel.eval()  # switch training off
            X, y = next(iter(X_test_loader))  # extract X,y from test dataloader
            with torch.no_grad():
                yHat = AnnModel(X)
            test_accuracy.append(100*torch.mean((torch.argmax(yHat, axis=1) == y).float()))

            # function output
        return train_accuracy, test_accuracy, losses_list

    def initialized_model(optimizer_choice, learning_rate, n_numbers, layers_numbers, momentum):
        FnnModel = createTheQwertyNet(n_numbers, layers_numbers)
        loss_function = nn.CrossEntropyLoss()
        optimizer_function = getattr(torch.optim, optimizer_choice)
        optimizer = optimizer_function(FnnModel.parameters(), lr=learning_rate, momentum=momentum)

        ## TESTE first the model work

        input_test = torch.rand(10, 2)
        print(FnnModel)
        print(FnnModel(input_test))


        return FnnModel, loss_function, optimizer


    def torch_transforme_data(data_np, label_np):

        data_torch = torch.tensor(data_np).float()
        label_torch = torch.tensor(label_np).long()
        X_train, X_test, y_train, y_test = train_test_split(data_torch, label_torch, train_size=0.9)

        my_batch_size = 16
        X_train_tensor = TensorDataset(X_train, y_train)
        X_test_tensor = TensorDataset(X_test, y_test)

        X_train_loader = DataLoader(X_train_tensor, batch_size=my_batch_size, shuffle=True, drop_last=True)
        X_test_loader = DataLoader(X_test_tensor, batch_size=X_test_tensor.tensors[0].shape[0])
        print(f'There are {len(X_train_loader)} batches, each with {my_batch_size} samples.')
        print(f'There are {len(X_test_loader)} batches, each with {my_batch_size} samples.')

        return X_train_loader, X_test_loader, data_torch, label_torch

    def create_data(numbers_obs):
        alpha_factor = 0.5
        A_center = [1, 1]
        B_center = [5, 1]
        C_center = [4, 4]
        STD_factor = 1

        A_catg = [A_center[0] + np.random.randn(numbers_obs) * STD_factor,
                  A_center[1] + np.random.randn(numbers_obs) * STD_factor]
        B_catg = [B_center[0] + np.random.randn(numbers_obs) * STD_factor,
                  B_center[1] + np.random.randn(numbers_obs) * STD_factor]
        C_catg = [C_center[0] + np.random.randn(numbers_obs) * STD_factor,
                  C_center[1] + np.random.randn(numbers_obs) * STD_factor]

        label_np = np.hstack((np.zeros((numbers_obs)), np.ones((numbers_obs)), 1 + np.ones((numbers_obs))))
        print("From label Np :", label_np.shape)  ## Columes Vector For all value

        # concatanate into a matrix
        data_np = np.hstack((A_catg, B_catg, C_catg)).T
        return data_np, label_np, alpha_factor

    numbers_obs = 50
    ## Variable for the Model ##

    optimizer_choice = "SGD"
    learning_rate = 0.01
    momentum = 0.95
    units_per_layer = 12
    layers_numbers = 4



    data_np, label_np, alpha_factor = create_data(numbers_obs)
    X_train_loader, X_test_loader, data_torch, label_torch = torch_transforme_data(data_np, label_np)
    plot_data(data_torch, label_torch, alpha_factor)
    FnnModel, loss_function, optimizer = initialized_model(optimizer_choice, learning_rate, units_per_layer, layers_numbers, momentum)
    train_accuracy, test_accuracy, losses_list = train_model(FnnModel, loss_function, optimizer, X_train_loader, X_test_loader)
    plot_train_result(train_accuracy, test_accuracy, losses_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_vs_depth()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

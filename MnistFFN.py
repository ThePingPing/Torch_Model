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



class FFNmnist(nn.Module):
    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(784, 64)

        ### hidden layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)

        ### output layer
        self.output = nn.Linear(32, 10)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))  # fully connected
        x = F.relu(self.fc2(x))
        return torch.log_softmax(self.output(x), axis=1)

def mnist_ffn_model():

    def scale_and_slip_data(data, scaler_factor):
        mnist_data_scale = data.drop(['label'], axis=1) / scaler_factor
        mnist_label = data["label"]
        print(mnist_data_scale)

        mnist_data_torch = torch.tensor(mnist_data_scale.values).float()

        mnist_label_torch = torch.tensor(mnist_label).long()

        X_train, X_test, y_train, y_test = train_test_split(mnist_data_torch, mnist_label_torch, train_size=0.9)

        # then convert them into PyTorch Datasets (note: already converted to tensors)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        batchsize = 32

        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

        return train_loader, test_loader

    def initialize_model(optimizer_choice, learning_rate, momentum):
        FfnModel = FFNmnist()
        loss_function = nn.NLLLoss()
        optimizer_function = getattr(torch.optim, optimizer_choice)
        optimizer = optimizer_function(FfnModel.parameters(), lr=learning_rate, momentum=momentum)
        return FfnModel, loss_function, optimizer

    def train_model(FfnModel, loss_function, optimizer, X_train_loader, X_test_loader):
        number_epochs = 60
        train_accuracy = []
        test_accuracy = []
        losses_list = torch.zeros(number_epochs)

        for epochis in range(number_epochs):
            FfnModel.train() ## turn on the train
            batch_accuracy = []
            batch_loss = []

            for X, y in X_train_loader:

                yHat = FfnModel(X)
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
            FfnModel.eval()  # switch training off
            X, y = next(iter(X_test_loader))  # extract X,y from test dataloader
            with torch.no_grad():
                yHat = FfnModel(X)
            test_accuracy.append(100*torch.mean((torch.argmax(yHat, axis=1) == y).float()))

            # function output
        return train_accuracy, test_accuracy, losses_list

    def plot_train_result(train_accuracy, test_accuracy, losses_list):

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(losses_list.detach())
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')

        ax[1].plot(train_accuracy, label='Train')
        ax[1].plot(test_accuracy, label='Test')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Accuracy')
        ax[1].set_title(f'Final model test accuracy: {test_accuracy[-1]:.2f}%')
        ax[1].legend()
        plt.show()

    # import dataset (comes with colab!)
    data = pd.read_csv('mnist_train.csv')

    scaler_factor = np.max(data.values)
    print(data.head())


    ## Variable for the Model ##

    optimizer_choice = "SGD"
    learning_rate = 0.01
    momentum = 0

    X_train_loader, X_test_loader = scale_and_slip_data(data, scaler_factor)
    FfnModel, loss_function, optimizer = initialize_model(optimizer_choice, learning_rate, momentum)
    train_accuracy, test_accuracy, losses_list = train_model(FfnModel, loss_function, optimizer, X_train_loader, X_test_loader)
    plot_train_result(train_accuracy, test_accuracy, losses_list)


if __name__ == '__main__':
    mnist_ffn_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

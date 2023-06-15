import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class AnnMultiModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2, 8)
        self.hidden = nn.Linear(8, 8)
        self.output = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        return self.output(x)



def ann_multi_class_model():

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

    def split_data(data_torch, label_torch, numberDataPoint):
        X_train, X_test , y_train, y_test = train_test_split(data_torch, label_torch, train_size=0.9)

        my_batch_size = 16
        X_train_tensor = TensorDataset(X_train, y_train)
        X_test_tensor = TensorDataset(X_test, y_test)

        X_train_loader = DataLoader(X_train_tensor, batch_size=my_batch_size, shuffle=True, drop_last=True)
        X_test_loader = DataLoader(X_test_tensor, batch_size=X_test_tensor.tensors[0].shape[0])
        print(f'There are {len(X_train_loader)} batches, each with {my_batch_size} samples.')
        print(f'There are {len(X_test_loader)} batches, each with {my_batch_size} samples.')

        return X_train_loader, X_test_loader

    def initialize_model(optimizer_choice, learning_rate,):
        AnnModel = AnnMultiModel()
        loss_function = nn.CrossEntropyLoss()
        optimizer_function = getattr(torch.optim, optimizer_choice)
        optimizer = optimizer_function(AnnModel.parameters(), lr=learning_rate)
        return AnnModel, loss_function, optimizer

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

    def plot_train_result(rain_accuracy, test_accuracy, losses_list):

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

    def plot_prediction_success(data_torch, label_torch, AnnModel):
        yHat = AnnModel(data_torch)
        predictions = torch.argmax(yHat, axis=1)
        accuracy = (predictions == label_torch).float()

        # compute overall accuracy
        total_accuracy = torch.mean(100 * accuracy).item()

        # and average by group
        accuracy_by_label = np.zeros(3)
        for i in range(3):
            accuracy_by_label[i] = 100 * torch.mean(accuracy[label_torch == i])

        plt.bar(range(3), accuracy_by_label)
        plt.ylim([80, 100])
        plt.xticks([0, 1, 2])
        plt.xlabel('Group')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Final accuracy = {total_accuracy:.2f}%')
        plt.show()

        colorShapes = ['bs', 'ko', 'g^']

        # show the data
        fig = plt.figure(figsize=(7, 7))

        # plot correct and incorrect labeled data
        for i in range(3):
            # plot all data points for this label
            plt.plot(data_torch[label_torch == i, 0], data_torch[label_torch == i, 1], colorShapes[i],
                     alpha=.3, label=f'Group {i}')

            # cross out the incorrect ones
            idxErr = (accuracy == 0) & (label_torch == i)
            plt.plot(data_torch[idxErr, 0], data_torch[idxErr, 1], 'rx')

        plt.title(f'The qwerties! ({total_accuracy:.0f}% accurately labeled)')
        plt.xlabel('qwerty dimension 1')
        plt.ylabel('qwerty dimension 2')
        plt.legend()
        plt.show()


    ## Variable for the Model ##

    optimizer_choice = "SGD"
    learning_rate = 0.01
    momentum = 0.95

    ## Build the Data for the model

    numberDataPoint = 300
    alpha_factor = 0.5
    A_center = [1, 1]
    B_center = [5, 1]
    C_center = [4, 4]
    STD_factor = 1

    A_catg = [A_center[0] + np.random.randn(numberDataPoint) * STD_factor,
              A_center[1] + np.random.randn(numberDataPoint) * STD_factor]
    B_catg = [B_center[0] + np.random.randn(numberDataPoint) * STD_factor,
              B_center[1] + np.random.randn(numberDataPoint) * STD_factor]
    C_catg = [C_center[0] + np.random.randn(numberDataPoint) * STD_factor,
              C_center[1] + np.random.randn(numberDataPoint) * STD_factor]

    label_np = np.hstack((np.zeros((numberDataPoint)), np.ones((numberDataPoint)), 1+np.ones((numberDataPoint))))
    print("From label Np :", label_np.shape)  ## Columes Vector For all value

    # concatanate into a matrix
    data_np = np.hstack((A_catg, B_catg, C_catg)).T
    print("Hello Data", data_np)

    ########## All Ready Start to Run the model ###########################################################################

    data_torch = torch.tensor(data_np).float()
    label_torch = torch.tensor(label_np).long()  ## take long format when you use CEE activation

    plot_data(data_torch, label_torch, alpha_factor)
    X_train_loader, X_test_loader = split_data(data_torch, label_torch, numberDataPoint)
    AnnModel, loss_function, optimizer = initialize_model(optimizer_choice, learning_rate, momentum)
    train_accuracy, test_accuracy, losses_list = train_model(AnnModel, loss_function, optimizer, X_train_loader, X_test_loader)
    plot_train_result(train_accuracy, test_accuracy, losses_list)
    plot_prediction_success(data_torch, label_torch, AnnModel)







if __name__ == '__main__':
    ann_multi_class_model()

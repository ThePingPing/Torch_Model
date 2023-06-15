
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class ANNwine(nn.Module):
    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(11, 16)

        ### hidden layers
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)

        ### output layer
        self.output = nn.Linear(32, 1)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))  # fully connected
        x = F.relu(self.fc2(x))
        return self.output(x)


def win_model():

    def normalized_data(win_data_pandas):
        data_features = win_data_pandas.drop(["quality"], axis=1)
        key_list = data_features.keys()

        for key in key_list:
            mean_key = np.mean(data_features[key])
            std_key = np.std(data_features[key], ddof=1)
            data_features[key] = (data_features[key]-mean_key)/std_key ## the Z normalization

        win_data_pandas.loc[win_data_pandas["quality"] < 6, "quality"] = 0
        win_data_pandas.loc[win_data_pandas["quality"] >= 6, "quality"] = 1
        data_label = pd.DataFrame(win_data_pandas["quality"][:len(data_features)], columns=['quality'])
        print(data_label.info)

        return data_features, data_label

    def split_data(win_data_norm, data_label):

        win_data_torch = torch.tensor(win_data_norm.values).float()
        win_label_torch = torch.tensor(data_label.values).float()

        # use scikitlearn to split the data
        X_train, X_test, y_train, y_test = train_test_split(win_data_torch, win_label_torch, train_size=0.9)

        # then convert them into PyTorch Datasets (note: already converted to tensors)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        # finally, translate into dataloader objects
        batchsize = 64

        train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

        return train_loader, test_loader

    def initialize_win_model():
        ANN_win_model = ANNwine()
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(ANN_win_model.parameters(), lr=0.01)

        return ANN_win_model, loss_function, optimizer

    def train_model(ANN_model, loss_function, optimizer, train_loader, test_loader):

        number_epochs = 500
        losses = np.zeros(number_epochs)
        train_accuracy = []
        test_accuracy = []

        for epochis in range(number_epochs):
            ANN_model.train()
            batch_accuracy = []
            batch_losses = []

            for X, y in train_loader:
                yHat = ANN_model(X)
                loss = loss_function(yHat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute training accuracy just for this batch
                batch_losses.append(loss.item())
                batch_accuracy.append(100 * torch.mean(((yHat > 0) == y).float()).item())

            train_accuracy.append(np.mean(batch_accuracy))
            losses[epochis] = np.mean(batch_losses)

            # test accuracy
            ANN_model.eval()  # switch training off
            X, y = next(iter(test_loader)) # extract X,y from test dataloader

            with torch.no_grad():
                yHat = ANN_model(X)

            test_accuracy.append(100 * torch.mean(((yHat > 0) == y).float()).item())

            # function output
        return train_accuracy, test_accuracy, losses, number_epochs

    def run_model(win_data_pandas):

        win_data_norm, data_label = normalized_data(win_data_pandas)
        train_loader, test_loader = split_data(win_data_norm, data_label)
        ANN_win_model, loss_function, optimizer = initialize_win_model()
        train_accuracy, test_accuracy, losses, number_epochs = train_model(ANN_win_model, loss_function, optimizer, train_loader, test_loader)

        fig = plt.figure(figsize=(10, 5))

        plt.plot((train_accuracy), 'bs-')
        plt.plot((test_accuracy), 'ro-')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend(['Train', 'Test'])
        plt.title('the Resulte is = %g' )

        plt.show()


    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    win_data_pandas = pd.read_csv(url, sep=';')
    # print(win_data_pandas.describe())
    # print(win_data_pandas.head())
    # print(win_data_pandas["quality"].unique())
    """win_data_torch = torch.tensor(win_data_pandas.drop(["quality"], axis=1).values).float()
    labels_torch = torch.tensor(win_data_pandas["quality"])
    print(win_data_torch)
    print(labels_torch.shape)
    print(labels_torch)"""

    ########## take the Normal Data ##############
    run_model(win_data_pandas)



if __name__ == '__main__':
    win_model()



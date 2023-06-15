import time

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



class XKModel(nn.Module):
    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(100, 100)

        ### hidden layers
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

        ### output layer
        self.output = nn.Linear(100, 2)

    # forward pass
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))  # fully connected
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)


def xavier_and_kaiming_model():

    def plot_collect_weights_biases(arr_weights, arr_biases):

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))

        ax[0].hist(arr_biases, 40)
        ax[0].set_title('The biases')

        ax[1].hist(arr_weights, 40)
        ax[1].set_title('The weights')

        # collect histogram data to show as line plots
        yB, xB = np.histogram(arr_biases, 30)
        yW, xW = np.histogram(arr_weights, 30)

        ax[2].plot((xB[1:] + xB[:-1]) / 2, yB / np.sum(yB), label='Bias')
        ax[2].plot((xW[1:] + xW[:-1]) / 2, yW / np.sum(yW), label='Weight')
        ax[2].set_title('Density estimate for both')
        ax[2].legend()

        # plot adjustments common to all subplots
        for i in range(3):
            ax[i].set_xlabel('Initial value')
            ax[i].set_ylabel('Count')
        ax[2].set_ylabel('Probability')

        plt.show()


    def change_default_weights(model):
        for params in model.named_parameters():
            if "weight" in params[0]:
                nn.init.xavier_normal_(params[1].data) ## it's inplace  But Now the Biases is in Kaiminig , and The weghts is in Xavier  --> that's just a exemple but
            # normaly put the same for both





    def collect_weights_biases(model):
        arr_weights = np.array([])
        arr_biases = np.array([])

        for params in model.named_parameters():
            if "bias" in params[0]:
                arr_biases = np.concatenate((arr_biases, params[1].data.numpy().flatten()), axis=0)
            elif "weight" in params[0]:
                arr_weights = np.concatenate((arr_weights, params[1].data.numpy().flatten()), axis=0)

        print(f'the number of weight is : {len(arr_weights)}')
        print(f'the number of biases is : {len(arr_biases)}')

        return arr_weights, arr_biases



    model1 = XKModel()
    model2 = XKModel()
    #print(model)
    arr_weights, arr_biases = collect_weights_biases(model1)
    plot_collect_weights_biases(arr_weights, arr_biases)
    change_default_weights(model2)
    arr_weights, arr_biases = collect_weights_biases(model2)
    plot_collect_weights_biases(arr_weights, arr_biases)

    weightvar = torch.var(model2.fc1.weight.data.flatten()).item() ## Take the Varience torch.var
    weightcount = len(model2.fc1.weight.data)

    # theoretical expected value
    sigma2 = 2 / (weightcount + weightcount)

    # drum rolllllll.....
    print('Theoretical sigma = ' + str(sigma2))
    print('Empirical variance = ' + str(weightvar))







if __name__ == '__main__':
    xavier_and_kaiming_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

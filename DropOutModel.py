import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class CreateModel(nn.Module):
    def __init__(self, drop_out_rate):
        super().__init__()

        self.input = nn.Linear(2, 128)
        self.hidden = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)

        self.dr = drop_out_rate

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training)  # training=self.training mean for the hidden layer
        # The model gonna train when he is in trainig mode
        # , and not trainig when he is in eval mode

        x = F.relu(self.hidden(x))
        x = F.dropout(x, p=self.dr, training=self.training)

        # for the output layer

        x = F.relu(self.output(x))

        return x


def drop_out_model():
    number_points = 200
    mult_factor = 3
    random_points = np.linspace(0, 4 * np.pi, number_points)

    r1_centre = 10
    r2_centre = 15

    A_group = [r1_centre * np.cos(random_points) + np.random.randn(number_points) * mult_factor,
               r1_centre * np.sin(random_points) + np.random.randn(number_points)]
    B_group = [r2_centre * np.cos(random_points) + np.random.randn(number_points),
               r2_centre * np.sin(random_points) + np.random.randn(number_points) * mult_factor]

    labels_np = np.vstack((np.zeros((number_points, 1)), np.ones((number_points, 1))))

    # concatanate into a matrix
    data_np = np.hstack((A_group, B_group)).T

    # convert to a pytorch tensor
    data_torch = torch.tensor(data_np).float()
    labels_torch = torch.tensor(labels_np).float()

    # show the data
    fig = plt.figure(figsize=(5, 5))
    plt.plot(data_torch[np.where(labels_torch == 0)[0], 0], data_torch[np.where(labels_torch == 0)[0], 1], 'bs')
    plt.plot(data_torch[np.where(labels_torch == 1)[0], 0], data_torch[np.where(labels_torch == 1)[0], 1], 'ko')
    plt.title("The qwerties' doughnuts!")
    plt.xlabel('qwerty dimension 1')
    plt.ylabel('qwerty dimension 2')
    plt.show()

    ########## Now Create The Spliting #############

    X_train, X_test, y_train, y_test = train_test_split(data_torch, labels_torch, train_size=0.8)

    X_train = TensorDataset(X_train, y_train)
    X_test = TensorDataset(X_test, y_test)

    batch_size = 16

    X_train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    X_test_loader = DataLoader(X_test, batch_size=X_test.tensors[0].shape[0])


    #################### just Try your Model #################


    def create_new_model(drop_out_rate):
        ANN_model = CreateModel(drop_out_rate)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(ANN_model.parameters(), lr=0.002)

        return ANN_model, loss_function, optimizer

    def train_model(ANN_model, loss_function, optimizer):

        number_epochs = 1000
        train_accuracy = []
        test_accuracy = []

        for epochis in range(number_epochs):
            ANN_model.train()
            batch_accuracy = []

            for X, y in X_train_loader:
                yHat = ANN_model(X)
                print(yHat.shape)
                print(y.shape)
                loss = loss_function(yHat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # compute training accuracy just for this batch
                batch_accuracy.append(100*torch.mean(((yHat > 0) == y).float()).item())

            train_accuracy.append(np.mean(batch_accuracy))

            # test accuracy
            ANN_model.eval()  # switch training off
            X, y = next(iter(X_test_loader))  # extract X,y from test dataloader
            yHat = ANN_model(X)
            test_accuracy.append(100 * torch.mean(((yHat > 0) == y).float()).item())

            # function output
        return train_accuracy, test_accuracy

    def run_model():
        drop_out_rate = 0.0
        ANN_model, loss_function, optimizer = create_new_model(drop_out_rate)
        train_accuracy, test_accuracy = train_model(ANN_model, loss_function, optimizer)

        fig = plt.figure(figsize=(10, 5))

        plt.plot(smooth(train_accuracy), 'bs-')
        plt.plot(smooth(test_accuracy), 'ro-')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend(['Train', 'Test'])
        plt.title('Dropout rate = %g' % drop_out_rate)

        plt.show()

    def smooth(x, k=5):
        return np.convolve(x, np.ones(k) / k, mode='same')

    def explore_model_drop_rate():
        drop_out_list = np.arange(10) / 10 ## list for differente probability to acction the drop out
        results = np.zeros((len(drop_out_list), 2))

        for dropOutIndex in range(len(drop_out_list)):
            ANN_model, loss_function, optimizer = create_new_model(drop_out_list[dropOutIndex])
            train_accuracy, test_accuracy = train_model(ANN_model, loss_function, optimizer)
            print(train_accuracy)

            results[dropOutIndex, 0] = np.mean(train_accuracy[-100 :]) ## take only the last 100 epochs result
            results[dropOutIndex, 1] = np.mean(test_accuracy[-100 :])
            print(results)


        final_plot(drop_out_list, results)

    def final_plot(drop_out_list, results):

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        ax[0].plot(drop_out_list, results, 'o-')
        ax[0].set_xlabel('Dropout proportion')
        ax[0].set_ylabel('Average accuracy')
        ax[0].legend(['Train', 'Test'])

        ax[1].plot(drop_out_list, -np.diff(results, axis=1), 'o-')
        ax[1].plot([0, .9], [0, 0], 'k--')
        ax[1].set_xlabel('Dropout proportion')
        ax[1].set_ylabel('Train-test difference (acc%)')

        plt.show()



    explore_model_drop_rate()


if __name__ == '__main__':
    drop_out_model()
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch.utils.data import DataLoader


def iris_cross_scikit_learn():

    def create_model(train_size):

        size_training_model = train_size

        ANNiris = nn.Sequential(

            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        lossfun = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.01)
        train_model(ANNiris, lossfun, optimizer, size_training_model)

    def train_model(ANNiris, lossfun, optimizer, size_training_model):
        numbers_epochs = 200
        #losses = torch.zeros(numbers_epochs)
        train_accuracy_value = []
        test_accuracy_value = []
        X_train, X_test, y_train, y_test = train_test_split(iris_data_torch, labels_torch,
                                                            train_size=size_training_model)
        #X_train = torch.utils.data.TensorDataset(X_train, y_train)
        #X_test = torch.utils.data.TensorDataset(X_test, y_test)

        #X_train_loader = DataLoader(X_train, shuffle=True, batch_size= 12)
        #X_test_loader = DataLoader(X_test, batch_size= X_test.tensors[0].shape[0])





        for epochi in range(numbers_epochs):

            yHat = ANNiris(X_train)
            loss = lossfun(yHat, y_train)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy_value.append(100*torch.mean((torch.argmax(yHat,axis=1) == y_train).float()).item())
            # test accuracy
            predlabels = torch.argmax(ANNiris(X_test), axis=1)
            test_accuracy_value.append(100 * torch.mean((predlabels == y_test).float()).item())
        plot_result(train_accuracy_value, test_accuracy_value, size_training_model)

    def plot_result(train_accuracy_value, test_accuracy_value, size_training_model):
        # plot the results
        print(size_training_model)
        fig = plt.figure(figsize=(10, 5))

        plt.plot(train_accuracy_value, 'ro-')
        plt.plot(test_accuracy_value, 'bs-')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend(['Train', 'Test'])
        plt.show()


    iris_data = sns.load_dataset("iris")
    iris_data_torch = torch.tensor(iris_data.drop(["species"], axis=1).values).float()
    labels_torch = torch.zeros(len(iris_data['species']),
                               dtype=torch.long)  ## VERY IMPORTANT DON"T FORGOT THE  dtype=torch.long
    labels_torch[iris_data.species == 'versicolor'] = 1
    labels_torch[iris_data.species == 'virginica'] = 2
    print(labels_torch.shape)

    trainSetSizes = np.linspace(.2, .95, 10)
    for i in range(len(trainSetSizes)):
        create_model(trainSetSizes[i])



if __name__ == '__main__':
    iris_cross_scikit_learn()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

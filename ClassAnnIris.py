
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F



class ANNiris(nn.Module):

  def __init__(self, number_units, number_layers):
    super().__init__()

    ## Dictionary to stock the layer for the model

    self.layers = nn.ModuleDict()
    self.number_layers = number_layers

    self.layers['input'] = nn.Linear(4, number_units)

    for i in range(number_layers):
        self.layers[f'hidden{i}'] = nn.Linear(number_units, number_units)

    self.layers['output'] = nn.Linear(number_units, 3)
    # forward pass

  def forward(self, x):
      # input layer (note: the code in the video omits the relu after this layer)
      x = F.relu(self.layers['input'](x))

      # hidden layers
      for i in range(self.number_layers):
          x = F.relu(self.layers[f'hidden{i}'](x))

      # return output layer
      x = self.layers['output'](x)
      return x

def initialized_model():

    def generate_data_iris():
        iris = sns.load_dataset("iris")
        print(iris.head())
        # sns.pairplot(iris, hue="species")
        print(iris.values)
        print("Hello number of element in colume", len(iris['species']))
        # plt.show()

        ############## Drop the Target Col From the Data set , is in DataFram Type so Covert to Tensor##############

        data_iris_torch = torch.tensor(iris.drop(["species"], axis=1).values).float()
        labels_torch = torch.zeros(len(iris['species']),
                                   dtype=torch.long)  ## VERY IMPORTANT DON"T FORGOT THE  dtype=torch.long
        labels_torch[iris.species == 'versicolor'] = 1
        labels_torch[iris.species == 'virginica'] = 2

        numbers_epochs = 2500
        learning_rate = 0.01
        units_for_layers = 12
        number_layers = 4
        my_model = ANNiris(units_for_layers, number_layers)
        print(my_model)

        loss_function = nn.CrossEntropyLoss()  ## use this loss if you have a multi class categorization
        optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)
        train_model(my_model, data_iris_torch, labels_torch, loss_function, optimizer, numbers_epochs)

    def train_model(ann_model, data_torch, labels_torch, loss_fun, opt, numbers_epochs):
        accuracy_value = []
        losses = torch.zeros(numbers_epochs)
        for epochi in range(numbers_epochs):
            ## Forward Propagation
            yHat = ann_model(data_torch)

            ## Compute loss
            loss = loss_fun(yHat, labels_torch)
            losses[epochi] = loss

            ## Back Propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            matches = torch.argmax(yHat, axis=1) == labels_torch  ## bool Vector
            numerical_matches = matches.float()  ## convert to num value ( 0/ 1)
            # print(numerical_matches)
            accuracy_perc = 100 * torch.mean(numerical_matches)
            accuracy_value.append(accuracy_perc)

        prediction_model(ann_model, data_torch, labels_torch)

    def prediction_model(ann_model, data_torch, labels_torch):
        predictions = ann_model(data_torch)
        prediction_label = torch.argmax(predictions, axis=1)
        total_accuracy = 100 * torch.mean((prediction_label == labels_torch).float())
        numbers_params = sum(p.numel() for p in ann_model.parameters() if p.requires_grad)
        print("The total Accuracy is :", total_accuracy, "The numbers of params is: ", numbers_params)

    generate_data_iris()


def np():
    number = np.arange(4, 101, 3)
    print(number)
if __name__ == '__main__':

    #initialized_model()
    np()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

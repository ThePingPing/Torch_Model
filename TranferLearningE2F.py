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

import sklearn.metrics as skm

from torchsummary import summary

import copy


def CreateTheEmnistModelConv(printtoggle=False):

    class emistnet(nn.Module):

        def __init__(self, printtoggle):
            super().__init__()

            """ for the Toggle """
            self.print = printtoggle

            ### ----------- Model Structure ------------ ###

            ##the Size output conv1 = np.floor((sizeImg + (2 * padding - Kernel_size) / stride_size ) +1) --> MaxPOOL

            self.conv1_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # output size: ((28+2*1-3)/1 + 1) / 2 = 28 /2 = 14, Start with 64 Kernels Map
            self.batch_normalize1 = nn.BatchNorm2d(64)

            self.conv2_layer = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)  # output size: ((14+2*0-3)/1 + 1) / 2 = 12 /2 = 6
            self.batch_normalize2 = nn.BatchNorm2d(128)

            self.conv3_layer = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)  # output size: ((6+2*0-3)/1 + 1) / 2 = 4 / 2 = 2
            self.batch_normalize3 = nn.BatchNorm2d(256)

            ### -------------- Linear Layers ------------ ###

            """ number_units_input = out_channels * (2 ** outputSize last convolution)"""

            number_units_input = 256 * (2**2)


            self.linear1 = nn.Linear(number_units_input, 256)
            self.linear2 = nn.Linear(256, 64)
            self.linear3 = nn.Linear(64, 26)

        def forward(self, x):

            if self.print: print(f'Input:{list(x.shape)}')

            # first block : convolution --> maxpooling --. batch normalization --> relu / leakyRelu

            x = F.max_pool2d(self.conv1_layer(x), 2)
            x = F.leaky_relu(self.batch_normalize1(x)) ## i decide to normalize before to change the varience
            # x = F.dropout(input=x, p=0.25, training=True, inplace=False)

            # Second block : convolution --> maxpooling --. batch normalization --> relu / leakyRelu

            x = F.max_pool2d(self.conv2_layer(x), 2)
            x = F.leaky_relu(self.batch_normalize2(x))
            # x = F.dropout(input=x, p=0.25, training=True, inplace=False) ## not work Good Maybe retry with more epoch But that's already not a good sign

            if self.print: print(f'Pass to Second Block:{list(x.shape)}')

            # Threes block : convolution --> maxpooling --. batch normalization --> relu / leakyRelu

            x = F.max_pool2d(self.conv3_layer(x), 2)
            x = F.leaky_relu(self.batch_normalize3(x))
            # x = F.dropout(input=x, p=0.25, training=True, inplace=False)

            if self.print: print(f'Pass to Threes Block:{list(x.shape)}')

            number_unites = x.shape.numel() / x.shape[0]
            x = x.view(-1, int(number_unites))

            if self.print: print(f' Fllaten to Vectors :{list(x.shape)}')

            ## ------- to Linear Layers --------- ##

            x = F.leaky_relu(self.linear1(x))
            # x = F.dropout(input=x, p=0.5, training=True, inplace=False)
            x = F.leaky_relu(self.linear2(x))
            # x = F.dropout(input=x, p=0.5, training=True, inplace=False)
            x = self.linear3(x)

            if self.print: print(f' After the linear Block  :{list(x.shape)}')

            return x

    ModelEmnist = emistnet(printtoggle)

    return ModelEmnist


def CnnEmnistModel():

    def create_image_data_emnist():

        data_emnist = torchvision.datasets.EMNIST(root="emnist", split="letters", download=True)

        # print(data_emnist.classes) ## print the Data Letters
        #
        # print(str(len(data_emnist.classes)) + "Classes" )
        # print(data_emnist.data.shape) ## check the shape


        """converte the torch.Size([124800, 28, 28]) for a tensor 4D , 1 chane so [124800, 1, 28, 28]l and of course int to float if not you can used in the model """

        data_images_emnist = data_emnist.data.view([124800, 1, 28, 28]).float()
        # print(data_images_emnist.shape)
        #print(data_images_emnist) ## Check the Matrixs images values

        """ Now check if N/A Categorie Have Data """

        # print(torch.sum(data_emnist.targets == 0)) ## ==> tensor(0) So you 0 Image with this label, you can Delete this features No Problem

        """ ==> tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26]) That a problem , coz the LOSS FUNCTION gonna convert to one hot incoding , so we have to start in 0 """
        # print(torch.unique(data_emnist.targets))


        """Remove The N/A classes , juste reindex the target """

        # print(data_emnist.class_to_idx)

        data_letters_categories = data_emnist.classes[1:]

        labels_emnist = copy.deepcopy(data_emnist.targets) - 1

        # print(torch.sum(labels_emnist == 0)) ## ==> check it's the A letters Target

        return data_images_emnist, data_letters_categories, labels_emnist


    def create_image_data_fmnist():

        data_fmnist = torchvision.datasets.FashionMNIST(root="fmnist", download=True)

        data_images_fmnist = data_fmnist.data.view([60000, 1, 28, 28]).float()
        labels_fmnist = copy.deepcopy(data_fmnist.targets)
        data_fashion_categories = data_fmnist.classes[:]

        # print("The classes is: ", data_fmnist.classes)
        # print("From fmsnit Shape ",  data_fmnist.data.shape)
        # print("From fmsnit image Shape ", data_images_fmnist.shape)
        # print(data_fashion_categories)


        return data_images_fmnist, data_fashion_categories, labels_fmnist


    def plot_values_images(data_images_emnist, data_images_fmnist):

        ######################## EMNIST VALUE PLOT ##############################

        plt.hist(data_images_emnist[:10, :, :, :].view(1, -1).detach(), 40);
        plt.title("Row Value Original EMNIST")
        plt.show() ## --> value beetwen 0, 255 --> you go to normalize the data image

        data_images_emnist /= torch.max(data_images_emnist)

        plt.hist(data_images_emnist[:10, :, :, :].view(1, -1).detach(), 40);
        plt.title("Row Value after Normalization EMNIST")
        plt.show()

        ######################## FMNIST VALUE PLOT ##############################

        plt.hist(data_images_fmnist[:10, :, :, :].view(1, -1).detach(), 40);
        plt.title("Row Value Original FMNIST")
        plt.show()  ## --> value beetwen 0, 255 --> you go to normalize the data image

        data_images_fmnist /= torch.max(data_images_fmnist)

        plt.hist(data_images_fmnist[:10, :, :, :].view(1, -1).detach(), 40);
        plt.title("Row Value after Normalization FMNIST")
        plt.show()


    def plot_some_images_emnist(data_images_emnist, data_letters_categories, labels_emnist):

        fig, axs = plt.subplots(3, 7, figsize=(14, 7))

        # print(data_images_emnist.shape[0])
        # print(np.random.randint(data_images_emnist.shape[0]))
        # print(data_letters_categories)
        # print(len(labels_emnist))

        for i, ax in enumerate(axs.flatten()):
            random_images = np.random.randint(data_images_emnist.shape[0])
            image_extract = np.squeeze(data_images_emnist[random_images, :, :])
            letter = data_letters_categories[labels_emnist[random_images]]

            ax.imshow(image_extract.T, cmap="gray")
            ax.set_title('The Letter "%s"' %letter)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()


    def plot_some_images_fmnist(data_images_fmnist, data_fashion_categories, labels_fmnist):

        fig, axs = plt.subplots(3, 7, figsize=(18, 7))

        # print(data_images_emnist.shape[0])
        # print(np.random.randint(data_images_emnist.shape[0]))
        # print(data_letters_categories)
        # print(len(labels_emnist))

        for i, ax in enumerate(axs.flatten()):
            random_images = np.random.randint(data_images_fmnist.shape[0])
            image_extract = np.squeeze(data_images_fmnist[random_images, :, :])
            fashion = data_fashion_categories[labels_fmnist[random_images]]

            ax.imshow(image_extract, cmap="gray")
            ax.set_title('Fashion "%s"' %fashion)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()


    def data_loader_convert(data_images_emnist, labels_emnist, data_images_fmnist, labels_fmnist):

        X_train, X_test, y_train, y_test = train_test_split(data_images_emnist, labels_emnist, train_size=0.9)
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(data_images_fmnist, labels_fmnist, train_size=0.9)

        # then convert them into PyTorch Datasets (note: already converted to tensors)
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)

        train_data_f = TensorDataset(X_train_f, y_train_f)
        test_data_f = TensorDataset(X_test_f, y_test_f)

        batchsize = 32

        train_loader_emnist = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader_emnist = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

        train_loader_fmnist = DataLoader(train_data_f, batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader_fmnist = DataLoader(test_data_f, batch_size=test_data.tensors[0].shape[0])



        print("Train loader EMNIST : ",  train_loader_emnist.dataset.tensors[0].shape)
        print("Test loader EMNIST : ", train_loader_emnist.dataset.tensors[1].shape)

        print("Train loader FMNIST : ", train_loader_fmnist.dataset.tensors[0].shape)
        print("Test loader FMNIST : ", train_loader_fmnist.dataset.tensors[1].shape)

        return train_loader_emnist, test_loader_emnist, train_loader_fmnist, test_loader_fmnist

    def initialize_model():

        ModelEmnist = CreateTheEmnistModelConv(False)
        loss_function = nn.CrossEntropyLoss() ## The Data not Normalized so Use BCE With Log
        learning_rate = (10 ** -3)
        optimizer = Adam(ModelEmnist.parameters(), lr=learning_rate)

        return ModelEmnist, loss_function, optimizer

    def test_model(ModelEmnist, loss_function, train_loader):

        X, y = next(iter(train_loader))
        yHat = ModelEmnist(X)
        loss = loss_function(yHat, torch.squeeze(y))

        print(" ")
        print("Yhat Shape", yHat.shape) ## ==> Yhat Shape torch.Size([32, 26]) It's Normal Coz batch size is 32 , and 26 letters
        print(" Thes loss ", loss)


    def train_model(Model, loss_function, optimizer, train_loader, test_loader, numbers_epochs):

        ## Implement a new model , never see any data


        ## Push to GPU
        # ModelEmnist.to(device="cuda")

        # number of epochs

        ## initialize losses
        train_Loss = torch.zeros(numbers_epochs)
        test_Loss = torch.zeros(numbers_epochs)
        train_error = torch.zeros(numbers_epochs)
        test_error = torch.zeros(numbers_epochs)

        ## loop over epochs
        for epochi in range(numbers_epochs):

            # loop over training data batches
            Model.train()
            batch_loss = []
            batch_error = []
            for X, y in train_loader:

                # push data to the GPU

                # X = X.to(device="cuda:0")
                # y = y.to(device="cuda:0")

                # forward pass and loss
                yHat = Model(X)
                loss = loss_function(yHat, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss from this batch
                batch_loss.append(loss.item())
                batch_error.append(torch.mean((torch.argmax(yHat, axis=1) != y).float()).item()) ## you can change to == if you prefere to see the Accuracy rate
            # end of batch loop...

            # and get average losses across the batches
            train_Loss[epochi] = np.mean(batch_loss)
            train_error[epochi] = 100 * np.mean(batch_error)

            # test accuracy
            Model.eval()
            X, y = next(iter(test_loader))  # extract X,y from test dataloader

            # push data to the GPU
            # X = X.to(device="cuda:0")
            # y = y.to(device="cuda:0")
            print("Im Still learn the Epoch Number is :", epochi)

            with torch.no_grad():  # deactivates autograd
                yHat = Model(X)
                loss = loss_function(yHat, y)

            test_Loss[epochi] = loss.item()
            test_error[epochi] = 100 * torch.mean((torch.argmax(yHat, axis=1) != y).float()).item()

        # end epochs
        ModelTrained = Model

        # function output
        return ModelTrained, train_Loss, test_Loss, train_error, test_error



    def plot_train_result(train_Loss, test_Loss, train_error, test_error):

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))

        ax[0].plot(train_Loss, 's-', label='Train loss')
        ax[0].plot(test_Loss, 'o-', label='Test loss')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_title('Losses')
        ax[0].legend()

        ax[1].plot(train_error, 's-', label='Train, Error')
        ax[1].plot(test_error, 'o-', label='Test Error, ')
        ax[1].set_ylabel('Error rate (%)')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Error')
        ax[1].set_title(f'Final model test Error: {test_error[-1]:.2f}%')
        ax[1].legend()
        plt.show()


    def plot_the_mistake_image_predicted(ModelEmnistTrained, test_loader, data_letters_categories):

        X, y = next(iter(test_loader))
        yHat = ModelEmnistTrained(X)

        ## Take a few random Exemples

        list_rand_ex = np.random.choice(len(y), size=21, replace=False)

        fig, axs = plt.subplots(3, 7, figsize=(14, 7))

        for i, ax in enumerate(axs.flatten()):

            image_extract = np.squeeze(X[list_rand_ex[i], 0, :, :])
            true_letters_cat = data_letters_categories[y[list_rand_ex[i]]]
            predict_letters_cat = data_letters_categories[torch.argmax(yHat[list_rand_ex[i], :])]
            color = "gray" if true_letters_cat == predict_letters_cat else "hot"

            ax.imshow(image_extract.T, cmap=color)
            ax.set_title(' Real L  %s, pred L %s' % (true_letters_cat, predict_letters_cat), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()


        ### And For the Final we look on the confusion Matrix to Know , witch letters not predict correctly ###

        confusion_matrix = skm.confusion_matrix(y, torch.argmax(yHat, axis=1), normalize="true")

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(confusion_matrix, "Blues", vmax=0.05)

        ## Laout Design ##

        plt.xticks(range(26), labels=data_letters_categories)
        plt.yticks(range(26), labels=data_letters_categories)
        plt.title("Confusion Matrix")
        plt.xlabel("Real Letters")
        plt.ylabel("Predicted Letters")

        plt.show()

    def test_fashion_on_ModelEmnist(ModelEmnistTrained, test_loader_fmnist):

        print("FROM TEST MODEL WITH FASHION TEST LOADER")

        X, y = next(iter(test_loader_fmnist))
        yHat = ModelEmnistTrained(X)

        accuracy_fashion = 100 * torch.mean((torch.argmax(yHat, axis=1) == y).float())
        print(f'ModelEmnistTrained performance on FASHION data: {accuracy_fashion:.2f}%')


    def transfer_learning_model(ModelEmnistTrained):

        ModelFmnist, loss_function, optimizer = initialize_model()

        """Reintialized all the target model weight from the Source model already Trained"""

        for target, source in zip(ModelFmnist.named_parameters(), ModelEmnistTrained.named_parameters()):
            target[1].data = copy.deepcopy(source[1].data)

        ModelFmnist.linear3 = nn.Linear(64, 10)

        print("From ModelFmnist parameters : ", ModelFmnist)

        return ModelFmnist, loss_function, optimizer

    def plot_the_mistake_image_predicted_fashion(ModelFmnistTrained, test_Loss_f, data_fashion_categories):

        X, y = next(iter(test_Loss_f))
        yHat = ModelFmnistTrained(X)

        ## Take a few random Exemples

        list_rand_ex = np.random.choice(len(y), size=21, replace=False)

        fig, axs = plt.subplots(3, 7, figsize=(24, 7))

        for i, ax in enumerate(axs.flatten()):

            image_extract = np.squeeze(X[list_rand_ex[i], 0, :, :])
            true_fashion_cat = data_fashion_categories[y[list_rand_ex[i]]]
            predict_fashion_cat = data_fashion_categories[torch.argmax(yHat[list_rand_ex[i], :])]
            color = "gray" if true_fashion_cat == predict_fashion_cat else "hot"

            ax.imshow(image_extract, cmap=color)
            ax.set_title(' Real f %s, pred f %s' % (true_fashion_cat, predict_fashion_cat), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()

        confusion_matrix = skm.confusion_matrix(y, torch.argmax(yHat, axis=1), normalize="true")

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(confusion_matrix, "Blues", vmax=0.05)

        ## Laout Design ##

        plt.xticks(range(10), labels=data_fashion_categories)
        plt.yticks(range(10), labels=data_fashion_categories)
        plt.title("Confusion Matrix")
        plt.xlabel("Real Fashion")
        plt.ylabel("Predicted Fashion")

        plt.show()


    ##################### MODEL STARTING #########################

    init_numbers_epochs = 8
    tranfer_numbers_epochs = 1

    data_images_emnist, data_letters_categories, labels_emnist = create_image_data_emnist()
    data_images_fmnist, data_fashion_categories, labels_fmnist = create_image_data_fmnist()
    plot_values_images(data_images_emnist, data_images_fmnist)
    plot_some_images_emnist(data_images_emnist, data_letters_categories, labels_emnist)
    plot_some_images_fmnist(data_images_fmnist, data_fashion_categories, labels_fmnist)
    train_loader_emnist, test_loader_emnist, train_loader_fmnist, test_loader_fmnist = data_loader_convert(data_images_emnist, labels_emnist, data_images_fmnist, labels_fmnist)
    ModelEmnist, loss_function, optimizer = initialize_model()
    test_model(ModelEmnist, loss_function, train_loader_emnist)
    ModelEmnistTrained, train_Loss, test_Loss, train_error, test_error = train_model(ModelEmnist, loss_function, optimizer, train_loader_emnist, test_loader_emnist, init_numbers_epochs)
    plot_train_result(train_Loss, test_Loss, train_error, test_error)
    plot_the_mistake_image_predicted(ModelEmnistTrained, test_loader_emnist, data_letters_categories)

    #################### NOW The TRANFER LEARNING ################################

    test_fashion_on_ModelEmnist(ModelEmnistTrained, test_loader_fmnist)
    ModelFmnist, loss_function, optimizer = transfer_learning_model(ModelEmnistTrained)
    ModelFmnistTrained, train_Loss_f, test_Loss_f, train_error_f, test_error_f = train_model(ModelFmnist, loss_function, optimizer, train_loader_fmnist, test_loader_fmnist, tranfer_numbers_epochs)
    plot_train_result(train_Loss_f, test_Loss_f, train_error_f, test_error_f)
    plot_the_mistake_image_predicted_fashion(ModelFmnistTrained, test_loader_fmnist, data_fashion_categories)




if __name__ == '__main__':
    CnnEmnistModel()
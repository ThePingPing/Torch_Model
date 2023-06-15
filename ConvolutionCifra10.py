import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt


def ConvolutionModel():

    def plot_cifra_img(cifra_data):
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))

        for ax in axs.flatten():
            """ Extract Img"""
            rand_index_img = np.random.choice(len(cifra_data.targets))
            img_choice = cifra_data.data[rand_index_img, :, :, :]
            img_label = cifra_data.classes[cifra_data.targets[rand_index_img]]

            """ Plot Img"""

            ax.imshow(img_choice)
            ax.text(16, 0, img_label, ha="center", fontweight="bold", color="k", backgroundcolor="y")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def resize_and_normalization(cifra_data):
        Ts = T.Compose([T.ToTensor(), T.Resize(32 * 4), T.Grayscale(num_output_channels=1)])

        # include the transform in the dataset
        cifra_data.transform = Ts

        img_original = cifra_data.data[123, :, :, :]

        # option 1a: apply the transform "externally" to an image
        img_gray_transform = Ts(cifra_data.data[123, :, :, :])

        # option 1b: use the embedded transform
        img_default_transform = cifra_data.transform(cifra_data.data[123, :, :, :])

        return img_original, img_gray_transform, img_default_transform

    def plot_transformation_img(img_original, img_tranformation, img_default_trans):
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        print(img_tranformation.shape)
        print(torch.squeeze(img_tranformation).shape)

        ax[0].imshow(img_original)
        ax[1].imshow(torch.squeeze(img_tranformation))
        ax[2].imshow(torch.squeeze(img_default_trans), cmap="gray")

        plt.show()







    cifra_data = torchvision.datasets.CIFAR10(root="cifra10", download=True)
    print(cifra_data)
    print(cifra_data.data.shape)
    print(cifra_data.classes)
    print(len(cifra_data.targets))
    plot_cifra_img(cifra_data)
    img_original, img_gray_transform, img_default_transform = resize_and_normalization(cifra_data)
    plot_transformation_img(img_original, img_gray_transform, img_default_transform )

if __name__ == '__main__':
    ConvolutionModel()

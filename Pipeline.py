# -*- coding: utf-8 -*-
# for better image plotting
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from neural_reg import NeuralNetwork_reg
from neural import NeuralNetwork
from Neuralnet import Pipeline


class Pipeline(self):

    def __init__(self, dataset='KMNIST', horizontal_flip=False, flip_percentage=0.5,
                   random_rotation=False, rotation_degree=20,   random_perspective=False, distortion_scale=0.6,
                   reg=False,  validation_size=0.2,  loss='cross', optimizer='adam',  epochs=5, lr_rate=0.002, batch_size=200):

        self.dataset = dataset
        self.horizontal_flip = horizontal_flip
        self.flip_percentage = flip_percentage
        self.reg = reg
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.lr_rate = lr_rate

        ### creating transform pipeline
        if horizontal_flip:
            self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(flip_percentage)])
        if random_rotation:
            self.transformations = transforms.Compose([transforms.RandomRotation(rotation_degree)])
        if random_perspective:
            self.transformations = transforms.Compose([transforms.RandomPerspective(distortion_scale)])

        self.transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))])

        self.Createdataset()
        self.Validation()
        self.Loader()
        self.Model()
        self.Loss()
        self.Optmizer()
        self.Train()

    def Createdataset(self):

        ### Downloading the training and test data depending on the dataset variable passed as a string
        if self.dataset == 'KMNIST':
            self.train_dataset = datasets.KMNIST('./data', download=True, train=True, transform=self.transformations)
            self.test_dataset = datasets.KMNIST('./data/', download=True, train=False, transform=self.transformations)
        elif self.dataset == 'MNIST':
            self.train_dataset = datasets.MNIST('./data', download=True, train=True, transform=self.transformations)
            self.test_dataset = datasets.MNIST('./data/', download=True, train=False, transform=self.transformations)
        elif self.dataset == 'FashionMNIST':
            self.train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=self.transformations)
            self.test_dataset = datasets.FashionMNIST('./data/', download=True, train=False, transform=self.transformations)

    def Validation(self):
        # Get the size of our train set
        training_size = len(self.train_dataset)
        # then, we create a list of indices from 0 to training size range
        indices = list(range(training_size))
        # Shuffling the indices
        np.random.shuffle(indices)
        # The shuffled index will split the validation and training datasets using numpy "floor" method:
        index_split = int(np.floor(training_size * validation_size))  # floor of the scalar `x` is the largest integer
        # Then, we get the training and validation set indices passing the index split
        validation_indices, training_indices = indices[:index_split], indices[index_split:]
        # Using SubsetRandomSampler we sample elements randomly from a list of indices
        self.training_sample = SubsetRandomSampler(training_indices)
        self.validation_sample = SubsetRandomSampler(validation_indices)


    def Loader(self):
        ### creating the data loader, passing the sampler created above
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.training_sample)
        self.valid_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.validation_sample)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size)


    def Model(self):
        ### defining the model with or without regularization
        if self.reg:  # if True the model will be instantiated with regularization (dropout)
            self.model = NeuralNetwork_reg()
        elif self.reg == False:  # if False, the model won't have regularization
            self.model = NeuralNetwork()

    def Loss(self):
        ### LOSS
        if self.loss == 'cross':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'poiss':
            self.criterion = nn.PoissonNLLLoss()
        elif self.loss == 'gauss':
            self.criterion = nn.GaussianNLLLoss()
        elif self.loss == 'nll':
            self.criterion = nn.NLLLoss()

    def Optmizer(self):
        ### OPTIMIZER
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_rate)
        if self.optimizer == 'adamax':
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr_rate)
        if optimizer == 'asgd':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=self.lr_rate)

    def Train(self):
        ### Training the NN
        train_losses, test_losses = [], []
        for e in range(self.epochs):
            running_loss = 0
            for image, label in self.train_loader:
                self.optimizer.zero_grad()
                log_ps = self.model(image)
                self.loss = self.criterion(log_ps, label)
                self.loss.backward()
                self.optimizer.step()
                running_loss += self.loss.item()
            else:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    self.model.eval()
                    for image, label in test_loader:
                        log_ps = self.model.forward(image)
                        prob = torch.exp(log_ps)
                        test_loss += criterion(log_ps, label)
                        k_prob, k_class = prob.topk(1, dim=1)
                        equals = k_class == label.view(*k_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                self.model.train()
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Training Loss: {:.3f}...".format(train_losses[-1]),
                  "Test Loss: {:.3f}...".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

        ### Showing the image and the probability barplot
        images, labels = next(iter(train_loader))

        img = images[0].view(1, 784)
        # gradient OFF
        with torch.no_grad():
            logps = self.model(img)

        # We need the exp to see the real predictions
        ps = torch.exp(logps)
        view_classify(img.view(1, 28, 28), ps, train_losses, test_losses, version=dataset)

        return (accuracy / len(test_loader)), train_losses, test_losses


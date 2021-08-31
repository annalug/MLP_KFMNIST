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
from neural import NeuralNetwork
from neural_reg import NeuralNetwork_reg



class Pipeline(dataset='KMNIST',
                   validation_size=0.2, reg=False, loss='cross', optimizer='adam',  # model parameters
                   batch_size=200, epochs=5, lr_rate=0.002):
    """This class builds and trains a neural network model.
    ---> dataset: downloads the training and test sets dependind on the dataset passed as a string.
         - options: 'KMNIST', 'MNIST', 'FashionMNIST'; ().
    ---> Transforms:
        1) horizontal_flip: True or False (default = False).
        - flip_percentage: probability of the image being flipped (default = 0.5).
        2) random_rotation: True or False (default = False).
        - rotation_degree: degree that the image will rotate (default = 20).
        3) random_perspective: True or False (default = False).
        - distortion_scale: argument to control the degree of distortion and ranges from 0 to 1 (default = 0.6).
   ---> Model, Loss and Optimizer:
        4) validation_size: size of the train samples that will be used to validate the model (default = 0.2).
        5) reg: True or False (default = False)| If the model will be defined with or without regularization.
        6) Loss functions: passed as a string. Options: "cross", "poiss", "gauss", "nll".
        7) Optimizer functions: passed as a string. Options: "adam", "adamax", "asgd".
   ---> Other parameters:
        8) batch_size: number of batches to pass into the dataloader (default = 200).
        9) epochs: defining the number of epochs (default = 5).
        10) lr_rate: the rate that the SGD will learn and reduce the loss function (default = 0.002).
        """
   def __init__(self, horizontal_flip=False, flip_percentage=0.5,
                random_rotation=False, rotation_degree=20,
                random_perspective=False, distortion_scale=0.6):
        ### creating transform pipeline
        if horizontal_flip:
            self.transformations = transforms.Compose([transforms.RandomHorizontalFlip(flip_percentage)])
        if random_rotation:
            self.transformations = transforms.Compose([transforms.RandomRotation(rotation_degree)])
        if random_perspective:
            self.transformations = transforms.Compose([transforms.RandomPerspective(distortion_scale)])

        self.transformations = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))])



    def create_dataset(self, dataset='KMNIST'):


        ### Downloading the training and test data dependind on the dataset variable passed as a string
        if self.dataset == 'KMNIST':
            self.train_dataset = datasets.dataset('./data', download=True, train=True, transform=self.transformations)
            test_dataset = datasets.dataset('./data/', download=True, train=False, transform=self.transformations)
        elif self.dataset == 'MNIST':
            train_dataset = datasets.MNIST('./data', download=True, train=True, transform=self.transformations)
            test_dataset = datasets.MNIST('./data/', download=True, train=False, transform=self.transformations)
        elif self.dataset == 'FashionMNIST':
            train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=self.transformations)
            test_dataset = datasets.FashionMNIST('./data/', download=True, train=False, transform=self.transformations)


    ### defining the model with or without regularization
    if reg:  # if True the model will be instantiated with regularization (dropout)
        model = NeuralNetwork_reg()
    elif reg == False:  # if False, the model won't have regularization
        model = NeuralNetwork()

    ### LOSS
    if loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    elif loss == 'poiss':
        criterion = nn.PoissonNLLLoss()
    elif loss == 'gauss':
        criterion = nn.GaussianNLLLoss()
    elif loss == 'nll':
        criterion = nn.NLLLoss()

    ### OPTIMIZER
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    if optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr_rate)
    if optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=lr_rate)


    def Validation(self):

        # Get the size of our train set
        training_size = len(train_dataset)

        # then, we create a list of indices from 0 to training size range
        indices = list(range(training_size))

        # Shuffling the indices
        np.random.shuffle(indices)

        # The shuffled index will split the validation and training datasets using numpy "floor" method:
        index_split = int(np.floor(training_size * validation_size))  # floor of the scalar `x` is the largest integer

        # Then, we get the training and validation set indices passing the index split
        validation_indices, training_indices = indices[:index_split], indices[index_split:]

        # Using SubsetRandomSampler we sample elements randomly from a list of indices
        training_sample = SubsetRandomSampler(training_indices)
        validation_sample = SubsetRandomSampler(validation_indices)

            ### creating the data loader, passing the sampler created above
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sample)
            valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=validation_sample)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


        ### Training the NN
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for image, label in train_loader:
            optimizer.zero_grad()
            log_ps = model(image)
            loss = criterion(log_ps, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                model.eval()
                for image, label in test_loader:
                    log_ps = model.forward(image)
                    prob = torch.exp(log_ps)
                    test_loss += criterion(log_ps, label)
                    k_prob, k_class = prob.topk(1, dim=1)
                    equals = k_class == label.view(*k_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            model.train()
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
        logps = model(img)

    # We need the exp to see the real predictions
    ps = torch.exp(logps)
    view_classify(img.view(1, 28, 28), ps, train_losses, test_losses, version=dataset)

    return (accuracy / len(test_loader)), train_losses, test_losses
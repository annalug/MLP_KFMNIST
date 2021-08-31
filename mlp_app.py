# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
from Pipeline import Pipeline
from neural_reg import NeuralNetwork_reg
from neural import NeuralNetwork

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Multilayer perceptron with MNIST,FMNIST and KMNIST')


optmizers = ['adamax','adam','asgd']
lr_rates = np.arange(0.001,0.01,0.003)
datasets = ['MNIST','FashionMNIST','KMNIST']

datasets = st.selectbox('Select the dataset', datasets)
lr_rates = st.selectbox('Select the learning rate', lr_rates)
optmizers = st.selectbox('Select the optmizer', optmizers)


model_nn = Pipeline(dataset=datasets,lr_rate=lr_rates, random_perspective=True, distortion_scale=0.8, optimizer=optmizers)

st.write('Image, classification and Train-Test losses')
model_nn.Graph()



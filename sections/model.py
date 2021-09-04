import numpy as np
from sections.Pipeline import Pipeline
import torch
import streamlit as st


__all__ = ["model"]

def model():
    st.title('Multilayer perceptron with MNIST,FMNIST and KMNIST')

    st.write('Using GPU: ', torch.cuda.is_available()) # checking if it's using GPU

    optmizers = ['adamax','adam','asgd']
    lr_rates = np.arange(0.001,0.01,0.003)
    datasets = ['MNIST','FashionMNIST','KMNIST']
    reg = ['True','False']

    datasets = st.selectbox('Select the dataset', datasets)
    lr_rates = st.selectbox('Select the learning rate', lr_rates)
    optmizers = st.selectbox('Select the optmizer', optmizers)
    regularization = st.selectbox('Regularization?', reg)

    model_nn= Pipeline(dataset=datasets,reg=regularization, lr_rate=lr_rates, random_perspective=True, distortion_scale=0.8, optimizer=optmizers)

    if st.checkbox('Train the model'):
        model_nn.Train()

    # if st.checkbox('Image, classification and Train-Test losses'):
    #     model_nn.Graph()

if __name__ == "__main__":
    model()

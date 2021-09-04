import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sections.Pipeline import Pipeline

__all__ = ["introd"]

def introd():

    st.title('Multilayer perceptron with MNIST,FMNIST and KMNIST')
    st.markdown('The field of artificial neural networks is often just called neural networks or multi-layer perceptrons after perhaps the most useful type of neural network. A perceptron is a single neuron model that was a precursor to larger neural networks.\
It is a field that investigates how simple models of biological brains can be used to solve difficult computational tasks like the predictive modeling tasks we see in machine learning. The goal is not to create realistic models of the brain, but instead to develop robust algorithms and data structures that we can use to model difficult problems.')
    st.markdown('font : https://machinelearningmastery.com/neural-networks-crash-course/')
    st.image('https://www.researchgate.net/profile/Allan-Kardec-Barros-Filho/publication/274240858/figure/fig1/AS:392021136166914@1470476530215/TOPOLOGY-OF-A-MULTILAYER-PERCEPTRON-NEURAL-NETWORK.png')

    st.title('The datasets')

    st.write('FMNIST')

    fig = plt.figure(figsize=(5, 5))
    rows = 4
    columns = 5

    fashion_classes = {0: 'T-shirt/top',
                       1: 'Trouser',
                       2: 'Pullover',
                       3: 'Dress',
                       4: 'Coat',
                       5: 'Sandal',
                       6: 'Shirt',
                       7: 'Sneaker',
                       8: 'Bag',
                       9: 'Ankle Boot'}
    model2 = Pipeline(dataset='FashionMNIST', reg=False, lr_rate=0.01, random_perspective=True,
                        distortion_scale=0.8, optimizer='adamax')
    images, labels = model2.Images()

    for idx in np.arange(20):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx].numpy().squeeze(), cmap='gray')
        ax.set_title(fashion_classes[labels[idx].item()], color='red')
        fig.tight_layout()
        st.pyplot()

if __name__ == "__main__":
    introd()
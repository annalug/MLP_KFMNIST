import numpy as np
import matplotlib.pyplot  as plt

## ADDING a a new graph
def view_classify(img, ps, train_losses, test_losses, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 6), nrows=1, ncols=3)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST" or "KMNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "FashionMNIST":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    # Graph of train and test losses
    ax3.plot(train_losses, label='Train Loss')
    ax3.plot(test_losses, label='Test Loss')
    ax3.legend()

    plt.tight_layout()
from torch import nn, optim
import torch.nn.functional as F



##### Neural Network ################
# defining the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # defining the layers and neurons
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # flattening the tensor
        x = x.view(x.shape[0], -1)
        # forward step
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

from abc import ABC
from typing import List

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataset

from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

torch.set_default_tensor_type('torch.DoubleTensor')


class NetworkArchitecture(nn.Module):
    def __init__(self):
        super(NetworkArchitecture, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(117, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc1(x)


class NeuralNetwork(nn.Module, ABC):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network_architecture: NetworkArchitecture = NetworkArchitecture()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network_architecture.parameters(), lr=0.001)
        self.loss: List = []

    def predict(self, inputs=None):
        return self.network_architecture(inputs)

    def fit(self, inputs=None, labels=None):
        self.optimizer.zero_grad()

        outputs = self.network_architecture(inputs)
        labels = labels.view(1, -1)
        labels = labels.to(torch.long)

        outputs = outputs.view(1, -1)

        loss = self.criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        self.loss.append(loss)

        self.optimizer.step()
        return loss


if __name__ == '__main__':

    with open('data/mushroom.csv', 'r') as f:
        trainset = f.read()
    clean = []
    for record in trainset.split('\n'):
        record_split = record.split(',')
        clean.append([record_split[1:], [record_split[0]]])

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    one_hot_encoder.fit([i[0] for i in clean])
    X = torch.from_numpy(one_hot_encoder.transform([i[0] for i in clean]).todense())
    one_hot_encoder.fit([i[1] for i in clean])
    y = torch.from_numpy(one_hot_encoder.transform([i[1] for i in clean]).todense())

    neural_network = NeuralNetwork()

    for i in enumerate(X):
        neural_network.fit(X[i[0]], y[i[0]])

    plt.plot(neural_network.loss)
    plt.show()

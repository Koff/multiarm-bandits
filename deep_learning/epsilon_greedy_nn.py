from typing import List
from matplotlib import pyplot as plt
import torch.utils.data.dataset

from sklearn.preprocessing import OneHotEncoder

from deep_learning.neural_network import NeuralNetwork


def main():
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
    # define constants
    epsilon: float = 0.05
    cumulative_reward: float = 0
    loss: List = []

    for i in enumerate(X):
        # Predict output
        predict_output = neural_network.predict(X[i[0]])
        _, predicted_class = predict_output[0].max(0)
        _, true_class = y[i[0]].max(0)

        if predicted_class == 0 and true_class == 0:
            cumulative_reward += 1

        elif predicted_class == 0 and true_class == 1:
            cumulative_reward -= 1
        new_loss = neural_network.fit(X[i[0]], y[i[0]])
        loss.append(new_loss)
        print("cumulative_reward: %s" % cumulative_reward)

    plt.plot(loss)
    plt.show()

if __name__ == "__main__":
    main()

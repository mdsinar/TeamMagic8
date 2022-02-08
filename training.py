import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import FloatTensor


def train(network, train_set, criterion, optimizer, epochs, test_set, verbose=False):
    # Trains a neural network
    acc = []
    test_acc = []

    for epoch in tqdm.trange(epochs):
        for x, y in train_set:
            optimizer.zero_grad()
            output = network.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        current_accuracy = get_accuracy(network, train_set)
        acc.append(current_accuracy)

        current_test_acc = get_accuracy(network, test_set)
        test_acc.append(current_test_acc)

        if epoch % 10 == 0 and verbose:
            print(f"Train Accuracy, Test Accuracy = {current_accuracy}, {current_test_acc}")

    return acc, test_acc


def get_accuracy(network, data):
    # tests the accuracy of the network on the dataset data and returns it (the accuracy)
    correct_count = 0
    all_count = 0

    with torch.no_grad():
        for x, y in data:
            outputs = network(x)
            _, pred = torch.max(outputs.data, 1)
            all_count += y.size(0)
            correct_count += (pred == y).sum().item()

    return correct_count / all_count


def plot_accuracy(epochs, train_acc, test_acc):
    t = np.arange(0, epochs)
    fig, ax = plt.subplots()
    plt.ylabel('Accuracy')
    plt.xlabel('Training time')
    ax.plot(t, train_acc, label='Train accuracy')
    ax.plot(t, test_acc, label='Test accuracy')
    legend = ax.legend(loc='lower right')
    legend.get_frame().set_facecolor('white')
    plt.show()


def get_iris_dataset(test_size=20):
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    input_shape = len(x[0])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    y_train = torch.as_tensor(y_train)
    y_test = torch.as_tensor(y_test)

    train_set = TensorDataset(FloatTensor(x_train), y_train)
    test_set = TensorDataset(FloatTensor(x_test), y_test)

    train_set = DataLoader(train_set, batch_size=10)
    test_set = DataLoader(test_set, batch_size=1)

    return train_set, test_set, input_shape


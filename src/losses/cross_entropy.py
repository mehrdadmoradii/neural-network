import numpy as np


def cross_entropy_loss(x, y):
    num_train = x.shape[0]

    softmax = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    cross_entropy = -np.log(softmax[range(num_train), y])

    loss = np.sum(cross_entropy)
    loss /= num_train

    dx = softmax.copy()
    dx[range(num_train), y] -= 1
    dx /= num_train

    return loss, dx

import numpy as np


class BinaryCrossEntropyLoss:

    name = "binary_cross_entropy"

    def __call__(self, x, y):
        num_train = x.shape[0]
        y = y.reshape((num_train, -1))
        a = 1 / (1 + np.exp(-x))
        loss = -y * np.log(a) - (1 - y) * np.log(1 - a)
        loss = np.sum(loss) / num_train

        dx = (-y / a) + ((1 - y) / (1 - a))
        dx = dx * (a * (1 - a))
        dx /= num_train

        return loss, dx

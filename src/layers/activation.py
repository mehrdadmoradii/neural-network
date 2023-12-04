import numpy as np

from .base import BaseLayer


def get_activation(name):
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    else:
        raise ValueError("Invalid activation name.")


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def forward(self, x):
        out = np.maximum(x, 0)
        self.cache = (x,)
        return out

    def backward(self, upstream_gradient):
        x = self.cache[0]
        dx = np.where(x > 0, upstream_gradient, 0)
        return dx


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.cache = None

    def build(self, input_shape):
        pass

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.cache = (x,)
        return out

    def backward(self, upstream_gradient):
        x = self.cache[0]
        sigmoid_x = 1 / (1 + np.exp(-x))
        local_gradient = sigmoid_x * (1 - sigmoid_x)
        dx = upstream_gradient * local_gradient
        return dx

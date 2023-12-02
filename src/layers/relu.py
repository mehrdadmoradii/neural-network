import numpy as np

from .base import BaseLayer


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

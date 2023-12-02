import numpy as np

from .base import BaseLayer


class Dense(BaseLayer):

    def __init__(self, units, activation=None, weight_scale=1e-3):
        super().__init__()
        self.units = units
        self.activation = activation
        self.weight_scale = weight_scale
        self.parameters = {"w": None, "b": None}
        self.gradients = {"w": None, "b": None}
        self.w = None
        self.b = None

    def reset_gradients(self):
        self.gradients["w"] = np.zeros_like(self.parameters["w"])
        self.gradients["b"] = np.zeros_like(self.parameters["b"])

    def build(self, x):
        input_dim = np.prod(x.shape[1:])
        self.parameters["w"] = np.random.normal(0.0, self.weight_scale, (input_dim, self.units))
        self.parameters["b"] = np.zeros(self.units)
        self.built = True

    def forward(self, x):
        x_flatten = x.reshape((x.shape[0], -1))
        y = x_flatten @ self.parameters["w"] + self.parameters["b"]
        if self.activation is not None:
            y = self.activation(y)
        self.cache = (x,)
        return y

    def backward(self, upstream_gradient):
        if self.activation is not None:
            upstream_gradient = self.activation.backward(upstream_gradient)
        x = self.cache[0]
        x_flatten = x.reshape((x.shape[0], -1))
        d_x_flatten = upstream_gradient.dot(self.parameters["w"].T)
        dx = d_x_flatten.reshape(x.shape)
        dw = x_flatten.T.dot(upstream_gradient)
        db = np.sum(upstream_gradient, axis=0)
        return dx, dw, db

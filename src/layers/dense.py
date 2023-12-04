import numpy as np

from .base import BaseLayer


class Dense(BaseLayer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.parameters = {"w": None, "b": None}
        self.gradients = {"w": None, "b": None}

    @property
    def w(self):
        return self.parameters["w"]

    @property
    def b(self):
        return self.parameters["b"]

    def weight_initializer(self, shape, gain=2.0):
        fan_in, fan_out = shape[0], shape[1]
        std_dev = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std_dev, size=shape)

    def reset_gradients(self):
        self.gradients["w"] = np.zeros_like(self.parameters["w"])
        self.gradients["b"] = np.zeros_like(self.parameters["b"])

    def build(self, x):
        input_dim = np.prod(x.shape[1:])
        self.parameters["w"] = self.weight_initializer((input_dim, self.units))
        self.parameters["b"] = np.zeros(self.units)
        self.gradients["w"] = np.zeros_like(self.parameters["w"])
        self.gradients["b"] = np.zeros_like(self.parameters["b"])
        self.built = True

    def forward(self, x):
        x_flatten = x.reshape((x.shape[0], -1))
        y = x_flatten @ self.w + self.b
        if self.activation is not None:
            y = self.activation(y)
        self.cache = (x,)
        return y

    def backward(self, upstream_gradient):
        if self.activation is not None:
            upstream_gradient = self.activation.backward(upstream_gradient)
        x = self.cache[0]
        x_flatten = x.reshape((x.shape[0], -1))
        d_x_flatten = upstream_gradient.dot(self.w.T)
        dx = d_x_flatten.reshape(x.shape)
        dw = x_flatten.T.dot(upstream_gradient)
        db = np.sum(upstream_gradient, axis=0)
        self.gradients["w"] += dw
        self.gradients["b"] += db
        return dx   # downstream gradient

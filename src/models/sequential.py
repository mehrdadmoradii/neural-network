import numpy as np

from src.layers.base import BaseLayer


def compiled_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.compiled:
            raise RuntimeError("Model is not compiled yet.")
        return func(self, *args, **kwargs)
    return wrapper


class Sequential:

    def __init__(self, layers: list[BaseLayer]):
        self.layers = layers
        self.compiled = False
        self.optimizer = None
        self.loss_function = None
        self.history = {"loss": [], "val_accuracy": []}

    def add(self, layer):
        if not isinstance(layer, BaseLayer):
            raise TypeError("Layer must be an instance of BaseLayer.")
        if self.compiled:
            raise RuntimeError("Model is already compiled.")
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = loss
        self.compiled = True

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def draw_batch(self, x, y, batch_size):
        num_samples = x.shape[0]
        indices = np.random.choice(num_samples, batch_size)
        return x[indices], y[indices]

    @compiled_required
    def forward(self, x, y):
        scores = self.predict(x)
        return self.loss_function(scores, y)

    @compiled_required
    def fit(self, x, y, x_val=None, y_val=None, epochs=1, batch_size=32, verbose=True):
        num_samples = x.shape[0]
        iterations_per_epoch = max(num_samples // batch_size, 1)

        for epoch in range(epochs):
            for i in range(iterations_per_epoch):
                self.reset_gradients()
                x_batch, y_batch = self.draw_batch(x, y, batch_size)
                loss, dloss = self.forward(x_batch, y_batch)
                self.backward(dloss)
                self.optimizer.update(self.layers, batch_size)

            self.history["loss"].append(loss)
            if x_val is not None and y_val is not None:
                accuracy = self.evaluate(x_val, y_val)
                self.history["val_accuracy"].append(accuracy)

            if verbose:
                output = f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}"
                if x_val is not None and y_val is not None:
                    accuracy = self.evaluate(x_val, y_val)
                    output += f" - val_accuracy: {accuracy:.4f}"
                print(output)

        return self.history

    def evaluate(self, x, y):
        if self.loss_function.name == "cross_entropy":
            num_samples = x.shape[0]
            scores = self.predict(x)
            return np.sum(np.argmax(scores, axis=1) == y) / num_samples
        # elif self.loss_function.name == "binary_cross_entropy":
        #     num_samples = x.shape[0]
        #     scores = self.predict(x)
        #     scores = 1 / (1 + np.exp(-scores))
        #     return np.sum(np.round(scores) == y.reshape(-1, 1)) / num_samples

    def backward(self, upstream_gradient):
        for layer in reversed(self.layers):
            upstream_gradient = layer.backward(upstream_gradient)

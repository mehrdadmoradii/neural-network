import abc

from typing import List

from src.layers.base import BaseLayer


class BaseOptimizer(abc.ABC):

    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def step(self, parameter, gradient):
        pass

    def update(self, layers: List[BaseLayer]):
        for layer in layers:
            if layer.parameters is None:
                continue
            for key, parameter in layer.parameters.items():
                gradient = layer.gradients[key]
                updated_parameter = self.step(parameter, gradient)
                layer.parameters[key] = updated_parameter

from .base import BaseOptimizer


class SGD(BaseOptimizer):

    def step(self, parameter, gradient):
        return parameter - self.learning_rate * gradient

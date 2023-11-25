import abc


class BaseLayer(abc.ABC):
    
    def __init__(self):
        self.cache = None
        self.built = False

    @abc.abstractmethod
    def build(self, input_shape):
        pass

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def backward(self, upstream_gradient):
        pass

    def __call__(self, *args, **kwargs):
        if not self.built:
            self.build(*args, **kwargs)
            self.built = True
        return self.forward(*args, **kwargs)

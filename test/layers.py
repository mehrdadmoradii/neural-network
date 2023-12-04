import unittest
import numpy as np

from src.layers import Dense
from src.layers.activation import ReLU, Sigmoid
from .utils import eval_numerical_gradient_array


class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        num_inputs = 2
        input_shape = (4, 5, 6)
        self.output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = self.output_dim * np.prod(input_shape)

        self.x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        self.w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), self.output_dim)
        self.b = np.linspace(-0.3, 0.1, num=self.output_dim)

        self.correct_out = np.array([
            [1.49834967, 1.70660132, 1.91485297],
            [3.25553199, 3.5141327, 3.77273342]])

    def test_build(self):
        layer = Dense(self.output_dim)
        self.assertTrue(layer.built is False)
        layer.build(self.x)
        self.assertTrue(layer.built is True)
        self.assertEqual(layer.parameters["w"].shape, self.w.shape)
        self.assertEqual(layer.parameters["b"].shape, self.b.shape)

    def test_forward(self):
        layer = Dense(self.output_dim)
        layer.build(self.x)
        layer.parameters["w"] = self.w
        layer.parameters["b"] = self.b
        out = layer(self.x)
        self.assertEqual(out.shape, (self.x.shape[0], self.output_dim))
        self.assertTrue(np.allclose(out, self.correct_out))

    def test_backward(self):
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        layer = Dense(5)
        layer.built = True
        layer.parameters["w"] = w
        layer.parameters["b"] = b
        layer.reset_gradients()

        def forward(x, w, b):
            return x.reshape((x.shape[0], -1)) @ w + b

        dx_num = eval_numerical_gradient_array(lambda x: forward(x, w, b), x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: forward(x, w, b), w, dout)
        db_num = eval_numerical_gradient_array(lambda b: forward(x, w, b), b, dout)

        _ = layer(x)
        dx = layer.backward(dout)
        dw, db = layer.gradients["w"], layer.gradients["b"]

        self.assertTrue(np.allclose(dx, dx_num))
        self.assertTrue(np.allclose(dw, dw_num))
        self.assertTrue(np.allclose(db, db_num))

    def test_backward_with_activation(self):
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)

        layer = Dense(5, activation=ReLU())
        layer.built = True
        layer.parameters["w"] = w
        layer.parameters["b"] = b
        layer.reset_gradients()

        def forward(x, w, b):
            out = x.reshape((x.shape[0], -1)) @ w + b
            return np.maximum(out, 0)

        dx_num = eval_numerical_gradient_array(lambda x: forward(x, w, b), x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: forward(x, w, b), w, dout)
        db_num = eval_numerical_gradient_array(lambda b: forward(x, w, b), b, dout)

        _ = layer(x)

        dx = layer.backward(dout)
        dw, db = layer.gradients["w"], layer.gradients["b"]

        self.assertTrue(np.allclose(dx, dx_num))
        self.assertTrue(np.allclose(dw, dw_num))
        self.assertTrue(np.allclose(db, db_num))


class TestReLULayer(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        self.correct_out = np.array([
            [0., 0., 0., 0.,],
            [0., 0., 0.04545455,  0.13636364,],
            [0.22727273,  0.31818182,  0.40909091,  0.5,]])

    def test_forward(self):
        layer = ReLU()
        out = layer(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(np.allclose(out, self.correct_out))

    def test_backward(self):
        dout = np.random.randn(*self.x.shape)
        layer = ReLU()
        _ = layer(self.x)
        dx = layer.backward(dout)
        dx_num = eval_numerical_gradient_array(lambda x: np.maximum(x, 0), self.x, dout)
        self.assertTrue(np.allclose(dx, dx_num))


class TestSigmoidLayer(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    def test_forward(self):
        layer = Sigmoid()
        out = layer.forward(self.x)
        expected = 1 / (1 + np.exp(-self.x))
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(np.allclose(out, expected))

    def test_backward(self):
        dout = np.random.randn(*self.x.shape)
        layer = Sigmoid()
        _ = layer.forward(self.x)
        dx = layer.backward(dout)
        dx_num = eval_numerical_gradient_array(lambda x: 1 / (1 + np.exp(-self.x)), self.x, dout)
        self.assertTrue(np.allclose(dx, dx_num))


if __name__ == '__main__':
    unittest.main()

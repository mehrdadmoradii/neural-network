import unittest

from src.optimizers import SGD

from src.layers import Dense


class TestSGD(unittest.TestCase):

    def test_step(self):
        learning_rate = 1e-2
        optimizer = SGD(learning_rate=learning_rate)
        parameter, gradient = 10, 2
        expected = parameter - learning_rate * gradient
        actual = optimizer.step(parameter, gradient)
        print(expected, actual)
        self.assertEqual(expected, actual)

    def test_update(self):
        learning_rate = 1e-2
        optimizer = SGD(learning_rate=learning_rate)
        pl1, gl1 = {"w": 10, "b": 20}, {"w": 2, "b": 3}
        pl2, gl2 = {"w": 30, "b": 40}, {"w": 4, "b": 5}

        layer1 = Dense(1)
        layer1.built = True
        layer1.parameters = pl1.copy()
        layer1.gradients = gl1.copy()

        layer2 = Dense(1)
        layer2.built = True
        layer2.parameters = pl2.copy()
        layer2.gradients = gl2.copy()

        layers = [layer1, layer2]

        optimizer.update(layers)

        self.assertEqual(layer1.parameters["w"], pl1["w"] - learning_rate * gl1["w"])
        self.assertEqual(layer1.parameters["b"], pl1["b"] - learning_rate * gl1["b"])
        self.assertEqual(layer2.parameters["w"], pl2["w"] - learning_rate * gl2["w"])
        self.assertEqual(layer2.parameters["b"], pl2["b"] - learning_rate * gl2["b"])


if __name__ == '__main__':
    unittest.main()

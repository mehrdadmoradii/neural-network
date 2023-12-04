import unittest

import numpy as np

from src.losses import CrossEntropyLoss
from .utils import eval_numerical_gradient


class TestCrossEntropyLoss(unittest.TestCase):

    def test_backward(self):
        num_classes, num_inputs = 10, 50
        x = 0.001 * np.random.randn(num_inputs, num_classes)
        y = np.random.randint(num_classes, size=num_inputs)

        loss = CrossEntropyLoss()

        dx_num = eval_numerical_gradient(lambda x: loss(x, y)[0], x, verbose=False)
        loss, dx = loss(x, y)

        self.assertTrue(np.allclose(dx_num, dx))


if __name__ == "__main__":
    unittest.main()

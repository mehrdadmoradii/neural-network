# Neural Network

Minimalistic deep learning framework developed from scratch, inspired by Keras and built upon the foundations of the CS231n assignments. Created for educational purposes, this lightweight framework provides a simple and intuitive API for understanding the intricacies of neural network construction and training.

## Getting Started

```python
from src.models import Sequential
from src.layers import Dense, ReLU
from src.optimizers import SGD
from src.losses import CrossEntropyLoss


model = Sequential([
    Dense(20, activation=ReLU()),
    Dense(20, activation=ReLU()),
    Dense(20, activation=ReLU()),
    Dense(3)
])

model.compile(
    loss=CrossEntropyLoss(),
    optimizer=SGD(learning_rate=5e-2),
)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, x_val=X_test, y_val=y_test)
```


### Models

- `Sequential`: Linear stack of layers for building the neural network.

### Layers

- `Dense`: Fully connected layer.
- `ReLU`: Positive part function.

### Optimizers

- `SGD`: Stochastic Gradient Descent optimizer.

### Losses

- `CrossEntropyLoss`: Cross-entropy loss for classification tasks.


## Contributing

Contributions are welcome! If you have ideas, suggestions, or find issues, please open a pull request. Your input is highly valued.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


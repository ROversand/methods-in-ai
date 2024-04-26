import numpy as np


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

# Neural Network Hyperparameters
n_hidden = 2
learning_rate = 0.001
epochs = 10000

# Initialising weights and biases
np.random.seed(0)
weights_hidden = np.random.randn(2, n_hidden) * 0.1
bias_hidden = np.zeros(n_hidden)
weights_output = np.random.randn(n_hidden) * 0.1
bias_output = np.zeros(1)

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the Neural Network
for epoch in range(epochs):
    # Forward pass
    z_hidden = np.dot(X_train, weights_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)
    z_output = np.dot(a_hidden, weights_output) + bias_output
    output = z_output  # Linear activation

    # Computes the loss (Mean Squared Error)
    loss = np.mean((y_train - output) ** 2)

    # Backward pass
    d_loss_output = (y_train - output).reshape(-1, 1)
    d_loss_weights_output = np.dot(a_hidden.T, d_loss_output)

    d_loss_a_hidden = np.dot(d_loss_output, weights_output.reshape(-1, 1).T)
    d_loss_weights_hidden = np.dot(X_train.T, sigmoid_derivative(a_hidden) * d_loss_a_hidden)

    # Updates weights and biases
    weights_output += learning_rate * d_loss_weights_output.flatten()
    bias_output += learning_rate * np.sum(d_loss_output, axis=0)
    weights_hidden += learning_rate * d_loss_weights_hidden
    bias_hidden += learning_rate * np.sum(d_loss_a_hidden * sigmoid_derivative(a_hidden), axis=0)

    # Prints loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Testing the Neural Network
z_hidden_test = np.dot(X_test, weights_hidden) + bias_hidden
a_hidden_test = sigmoid(z_hidden_test)
z_output_test = np.dot(a_hidden_test, weights_output) + bias_output
output_test = z_output_test  # Linear activation

# Calculates MSE for testing data
test_loss = np.mean((y_test - output_test) ** 2)
print(f'Test Loss: {test_loss}')

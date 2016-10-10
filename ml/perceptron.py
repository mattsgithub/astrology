import numpy as np


class Perceptron(object):
    """
        Batch gradient descent
        implementation of the simplest
        neural network

        TODO:
        [ ] Vetorize implementation
    """

    def __init__(self, p, M, eta):
        # Initialize weights
        self._w = np.random.random(p+1)
        self._M = M
        self._eta = eta

    def _transfer_function(self, w, x):
        return w.dot(x)

    def _activation_function(self, x):
        return 1. if x > 0 else 0.

    def train(self, X, y):
        # Prepend column of ones to matrix X
        N = X.shape[0]
        ones = np.ones(N)
        X = np.column_stack((ones, X))

        # For each epoch
        for e in range(self._M):
            # Update each weight
            for j, w in enumerate(self._w):
                sum_ = 0.0
                for i, x in enumerate(X):
                    r = self._transfer_function(self._w, x)
                    y_hat = self._activation_function(r)
                    sum_ += (y_hat - y[i]) * X[i][j]
                w = w - (self._eta/N) * sum_

    def predict(self, x):
        r = self._transfer_function(self._w, x)
        r = self._activation_function(r)
        return int(r)

X = np.array([[1, 1], [-1, -1]])
y = np.array([1, 0])
p = Perceptron(2, 10, 0.2)
p.train(X, y)

x = np.array([1., -2, -2])
print p.predict(x)

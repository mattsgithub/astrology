import numpy as np
from matplotlib import pyplot as plt


class Perceptron(object):
    """
        Batch gradient descent
        implementation of the simplest
        neural network

        TODO:
        [ ] Vetorize implementation
    """

    def __init__(self, num_epochs, step_size):
        self._num_epochs = num_epochs
        self._step_size = step_size
        self._X = None
        self._y = None
        self._w = None
        self._N = None
        self._p = None

    def _transfer_function(self, w, x):
        return w.dot(x)

    def _activation_function(self, x):
        return 1. if x >= 0 else 0.

    def train(self, X, y):

        # Number of training examples
        self._N = X.shape[0]

        # Number of features
        self._p = X.shape[1]

        # Number of weights (add one for bias)
        self._w = np.random.random(self._p+1)

        # Design matrix (prepend column of ones)
        self._X = np.column_stack((np.ones(self._N), X))

        # Ground truth
        self._y = y

        # For each epoch ...
        for e in range(self._num_epochs):
            # For each weight ...
            for j, w in enumerate(self._w):
                sum_ = 0.0
                for i, x in enumerate(self._X):
                    r = self._transfer_function(self._w, x)
                    y_hat = self._activation_function(r)
                    sum_ += (y_hat - y[i]) * self._X[i][j]
                w = w - (self._step_size/self._N) * sum_

    def predict(self, x):
        r = self._activation_function(
                self._transfer_function(self._w, x)
                )
        return int(r)

    def plot(self):
        if self._p != 2:
            raise Exception('Dimensions must be 2')

        # Plot points
        for i, x in enumerate(self._X):
            color = 'green' if self._y[i] == 1 else 'red'
            plt.scatter(x[1], x[2], color=color)

        # Plot boundary
        col = self._X[:, 1]
        x = np.arange(col.min(), col.max(), .1)
        y = [(self._w[0] + i * self._w[1])/(self._w[2] * -1.) for i in x]
        plt.plot(x, y, '--', color='lightgray')
        plt.title('Percepton\nEpochs: {0}, Step Size: '
                  '{1}'.format(self._num_epochs, self._step_size))
        plt.show()

    @staticmethod
    def demo():
        p = Perceptron(num_epochs=50, step_size=0.2)

        # Number of positive examples
        num_pos = 100

        # Number of negative examples
        num_neg = 100

        # Covariance matrix
        cov = [[1., 0], [0., 1.]]

        # Mean vector for positive
        # and negative examples
        mu_pos = [2., 2.]
        mu_neg = [-2., -2.]

        # Response vector
        y = [1.]*num_pos + [0.]*num_neg

        # Design matrix
        X1 = np.random.multivariate_normal(mean=mu_pos,
                                           cov=cov,
                                           size=num_pos)

        X2 = np.random.multivariate_normal(mean=mu_neg,
                                           cov=cov,
                                           size=num_neg)
        # Combine both into one matrix
        X = np.concatenate((X1, X2))

        # Train perceptron
        p.train(X, y)

        # Plot classifier
        p.plot()

Perceptron.demo()

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork(object):

    def __init__(self):
        self._h = None
        self._W1 = None
        self._W2 = None

    def predict(self, x):
        return self._forward_propagate(x)

    def train(self, X, y):
        num_hidden = 3
        num_output = 1
        num_epochs = 20
        num_inputs = 2
        num_examples = len(y)

        eta = 0.2

        self._W1 = np.random.rand(num_inputs + 1, num_hidden)
        self._W2 = np.random.rand(num_hidden, num_output)

        for num_epochs in xrange(num_epochs):
            # Randomize for each pass; Used for stochastic gradient descent
            for n in xrange(num_examples):
                # First, make prediction for this
                # example
                y_hat = self._forward_propagate(X[n])

                # Compute first error terms
                e1 = np.zeros(num_output)
                for k in xrange(num_output):
                    e1[k] = y_hat[k] * (1. - y_hat[k]) * (y_hat[k] - y[k])

                # For each weight....
                for k in xrange(num_output):
                    for j in xrange(num_hidden):
                        self._W2[j][k] = self._W2[j][k] - eta * e1[k] * self._h[j]

                # Compute second layer of error terms
                e2 = np.zeros(num_hidden)
                for j in xrange(num_hidden):
                    e2[j] = self._h[j] * (1. - self._h[j]) * np.sum([e1[k] * self._W[j][k] for k in xrange(num_output)])

                # For each weight....
                for j in xrange(num_hidden):
                    for i in xrange(num_inputs + 1):
                        self._W1[i][j] = self._W1[i][j] - eta * e2[j] * X[n][i]

    def _forward_propagate(self, x):
        self._h = sigmoid(self._W1.dot(x))
        y_hat = sigmoid(self._W2.dot(self._h))
        return y_hat

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


ann = NeuralNetwork()
ann.train(X, y)

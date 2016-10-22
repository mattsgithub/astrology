import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork(object):

    def __init__(self):
        self._h = None
        self._W1 = None
        self._W2 = None

    def predict(self, x):
        x = [1.] + x
        return self._forward_propagate(x)

    def train(self,
              X,
              y,
              num_inputs=1,
              num_hidden=6,
              num_output=1,
              eta=0.2,
              num_epochs=1000):

        num_examples = len(y)

        # Design matrix (prepend column of ones)
        X = np.column_stack((np.ones(num_examples), X))

        # Add one for bias unit
        self._W1 = np.random.rand(num_hidden, num_inputs + 1)
        self._W2 = np.random.rand(num_output, num_hidden)

        for num_epochs in xrange(num_epochs):
            # Randomize for each pass; Used for stochastic gradient descent
            for n in xrange(num_examples):
                # First, make prediction for this
                # example
                y_hat = self._forward_propagate(X[n])

                # Compute first error terms
                e1 = np.zeros(num_output)
                for k in xrange(num_output):
                    e1[k] = y_hat[k] * (1. - y_hat[k]) * (y_hat[k] - y[n][k])

                # For each weight....
                for k in xrange(num_output):
                    for j in xrange(num_hidden):
                        change = self._W2[k][j] - eta * e1[k] * self._h[j]
                        self._W2[k][j] = change

                # Compute second layer of error terms
                e2 = np.zeros(num_hidden)
                for j in xrange(num_hidden):
                    len_ = xrange(num_output)
                    _sum = np.sum([e1[k] * self._W1[j][k] for k in len_])
                    e2[j] = self._h[j] * (1. - self._h[j]) * _sum

                # For each weight....
                for j in xrange(num_hidden):
                    for i in xrange(num_inputs + 1):
                        self._W1[j][i] = self._W1[j][i] - eta * e2[j] * X[n][i]

    def _forward_propagate(self, x):
        u1 = self._W1.dot(x)
        self._h = sigmoid(u1)

        u2 = self._W2.dot(self._h)
        y_hat = sigmoid(u2)

        return y_hat

    @staticmethod
    def demo():
        ann = NeuralNetwork()

        # Training domain
        X_train = np.linspace(start=0., stop=2. * np.pi, num=40)
        y_train = [[y] for y in .5 * (np.sin(X_train) + 1.)]

        # Train network
        ann.train(X_train, y_train)

        # Plot ground truth
        plt.plot(X_train, y_train, color='gray', marker='x')

        # Plot prediction
        plt.scatter(X_train, [ann.predict([i]) for i in X_train], marker='x', color='green')

        plt.show()

if __name__ == '__main__':
    NeuralNetwork.demo()

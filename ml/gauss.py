import numpy as np
from numpy.linalg import inv, det
from matplotlib import pyplot as plt

from astrology import util


def get_gauss_pdf_value(x, mu, cov):
    p = len(mu)
    xs = x - mu
    covi = inv(cov)
    arg = -0.5 * (xs.T).dot(covi.dot(xs))

    # Normalization constant
    C = (((2.0 * np.pi)**p)*det(cov))**(-0.5)

    prob = C * np.exp(arg)
    return prob


class QDA(object):
    def __init__(self):

        # Holds all of the class labels
        self._classes = set()

        # The number of features
        self._p = None

        # The number of examples by class
        self._N_by_class = dict()

        # The total number of examples
        self._N = 0

        # Holds each mean vector, accessible by class
        self._mean_vector_by_class = dict()

        # Holds each covariance matrix, accessible by class
        self._cov_matrix_by_class = dict()

        # Holds each prior by class
        self._prior_by_class = dict()

    def predict(self, x):
        """predict

          Parameters
          ----------
            x : np.array
                Predict the class from this feature vector

          Returns
          -------
          x,y : str, dict
        """

        # Calculate p(y)p(x|y) for each class
        pred = {y: self._prior_by_class[y] *
                get_gauss_pdf_value(x,
                self._mean_vector_by_class[y],
                self._cov_matrix_by_class)
                for y in self._classes}

        # Normalize
        N = float(sum(pred.values()))
        pred = {y: pred[y]/N for y in pred}

        # Get max value
        pred_class = max(pred.iterkeys(), key=(lambda k: pred[k]))

        # Return result
        return pred_class, pred

    def observe(self, x, y):
        """Make an observation of a labeled
           data pointe

            Parameters
            ----------
            x : numpy.array
                A feature vector

            y : Object
                The class label associated
                with this feature vector
        """
        # If none, then this is the first
        # time making an observation
        if self._p is None:
            self._p = len(x)
        elif self._p != len(x):
            raise ValueError("Dimension does not match previous dimensions")

        # Is this a new class?
        if y not in self._classes:
            self._instantiate_class(y)

        # Update total number of observations
        self._N += 1.0

        # Update number of observations by class
        self._N_by_class[y] += 1.0

        N = self._N_by_class[y]
        mu = self._mean_vector_by_class[y]
        cov = self._cov_matrix_by_class[y]

        # Compute updated covariance matrix
        cov = util.get_updated_cov(cov, mu, N, x)

        # Compute updated mean vector
        mu = util.get_updated_mean(mu, N, x)

        # Compute updated prior
        prior = self._N_by_class[y] / self._N

        # Replace current cov, mean, and prior
        self._mean_vector_by_class[y] = mu
        self._cov_matrix_by_class[y] = cov
        self._prior_by_class[y] = prior

    def _instantiate_class(self, y):
        """ Creates dictionary entries
            for this new class in order to
            record various stats

            Parameters
            ----------
            y : Object
                The class label

        """

        # Add to class set
        self._classes.add(y)

        # Initialize number of training examples
        # by class
        self._N_by_class[y] = 0.0

        # Initialize mean by class to zeros
        self._mean_vector_by_class[y] = np.zeros(self._p)

        # Initialize covariance matrix by class to zeros
        s = (self._p, self._p)
        self._cov_matrix_by_class[y] = np.zeros(s)

        # Initialize prior by class
        self._prior_by_class[y] = 0.0

    @staticmethod
    def demo():
        """ Demonstrates how QDA works
        """

        # Create 2D Gaussian for class 1
        N1 = 100
        mean = np.array([3., 3.])
        cov = np.array([[5.0, 0.], [0., 5.0]])
        x1, y1 = np.random.multivariate_normal(mean, cov, N1).T
        plt.plot(x1, y1, 'o', color='red')

        # Create 2D Gaussian for class 2
        mean = np.array([5., 5.])
        cov = np.array([[1., 0.], [0., 1.]])
        N2 = 100
        x2, y2 = np.random.multivariate_normal(mean, cov, N2).T
        plt.plot(x2, y2, 'o', color='blue')

        qda = QDA()

        # Make observations for each class
        for i in xrange(N1):
            x = np.array([x1[i], y1[i]])
            qda.observe(x, 'c1')

        for i in xrange(N2):
            x = np.array([x2[i], y2[i]])
            qda.observe(x, 'c2')

        # Now, make 100 predictions
        for i in xrange(100):
            x1 = np.random.random()*10.0
            x2 = np.random.random()*10.0
            x = np.array([x1, x2])
            y, r = qda.predict(x)
            color = 'pink' if y == 'c1' else 'lightblue'
            plt.plot(x1, x2, 'o', color=color)

        plt.axis('equal')
        plt.show()


class LDA(object):
    def __init__(self):
        self._classes = set()
        self._p = None
        self._N_by_class = dict()
        self._N = 0
        self._prior_by_class = dict()
        self._mean_vector_by_class = dict()
        self._cov_matrix = None

    def predict(self, x):
        """predict

          Parameters
          ----------
            x : np.array
                Predict the class from this feature vector

          Returns
          -------
          x,y : str, dict
        """

        # Calculate p(y)p(x|y) for each class
        pred = {y: self._prior_by_class[y] *
                get_gauss_pdf_value(x,
                self._mean_vector_by_class[y],
                self._cov_matrix)
                for y in self._classes}

        # Normalize
        N = float(sum(pred.values()))
        pred = {y: pred[y]/N for y in pred}

        # Get max value
        pred_class = max(pred.iterkeys(), key=(lambda k: pred[k]))

        # Return result
        return pred_class, pred

    def observe(self, x, y):
        """Make an observation of a labeled
           data pointe

            Parameters
            ----------
            x : numpy.array
                A feature vector

            y : Object
                The class label associated
                with this feature vector
        """
        # If none, then this is the first
        # time making an observation
        if self._p is None:
            self._p = len(x)
        elif self._p != len(x):
            raise ValueError("Dimension does not match previous dimensions")

        # Is this a new class?
        if y not in self._classes:
            self._instantiate_class(y)

        # Update total number of observations
        self._N += 1.0

        # Update number of observations by class
        self._N_by_class[y] += 1.0

        N = self._N_by_class[y]
        mu = self._mean_vector_by_class[y]
        cov = self._cov_matrix

        # Compute updated covariance matrix
        cov = util.get_updated_cov(cov, mu, self._N, x)

        # Compute updated mean vector
        mu = util.get_updated_mean(mu, N, x)

        # Compute updated prior
        prior = self._N_by_class[y] / self._N

        # Replace current cov, mean, and prior
        self._mean_vector_by_class[y] = mu
        self._cov_matrix = cov
        self._prior_by_class[y] = prior

    def _instantiate_class(self, y):
        """ Creates dictionary entries
            for this new class in order to
            record various stats

            Parameters
            ----------
            y : Object
                The class label

        """

        # Add to class set
        self._classes.add(y)

        # Initialize number of training examples
        # by class
        self._N_by_class[y] = 0.0

        # Initialize mean by class to zeros
        self._mean_vector_by_class[y] = np.zeros(self._p)

        # Initialize covariance matrix to zeros
        s = (self._p, self._p)
        self._cov_matrix = np.zeros(s)

        # Initialize prior by class
        self._prior_by_class[y] = 0.0

    @staticmethod
    def demo():
        # Issue on mac os x:
        # Run using 'frameworkpython' instead of 'python'
        #
        # Generate points from true
        # underly distribution

        # Class 1
        mean = np.array([3., 3.])
        cov = np.array([[1., 0.], [0., 1.]])
        N1 = 100
        x1, y1 = np.random.multivariate_normal(mean, cov, N1).T
        plt.plot(x1, y1, 'o', color='red')

        # Class 2
        mean = np.array([8., 8.])
        cov = np.array([[1., 0.], [0., 1.]])
        N2 = 100
        x2, y2 = np.random.multivariate_normal(mean, cov, N2).T
        plt.plot(x2, y2, 'o', color='blue')

        x = np.array([x1, x2])
        y = np.array([y1, y2])

        x_min = np.amin(x)
        x_max = np.amax(x)
        y_min = np.amin(y)
        y_max = np.amax(y)

        # Run LDA
        lda = LDA()
        for i in xrange(N1):
            x = np.array([x1[i], y1[i]])
            lda.observe(x, 'c1')

        for i in xrange(N2):
            x = np.array([x2[i], y2[i]])
            lda.observe(x, 'c2')

        for i in xrange(100):
            x1 = np.random.random()*10.0
            x2 = np.random.random()*10.0
            x = np.array([x1, x2])
            y, r = lda.predict(x)
            color = 'pink' if y == 'c1' else 'lightblue'
            plt.plot(x1, x2, 'o', color=color)

        # Plot decision boundary
        cov = lda._cov_matrix
        covi = inv(cov)
        mu1, mu2 = lda._mean_vector_by_class.values()
        pi1, pi2 = lda._prior_by_class.values()
        x = [i for i in np.arange(-5.0, 15.0)]
        y = [LDA.get_y(xi, mu1, mu2, pi1, pi2, covi) for xi in x]
        plt.plot(x, y, color="black")

        margin = 1.0
        plt.xlim([x_min - margin, x_max + margin])
        plt.ylim([y_min - margin, y_max + margin])
        plt.show()

    @staticmethod
    def get_y(x, mu1, mu2, pi1, pi2, covi):
        beta1 = covi.dot(mu1)
        beta2 = covi.dot(mu2)
        gamma1 = -0.5 * (mu1).dot(covi).dot(mu1) + np.log(pi1)
        gamma2 = -0.5 * (mu2).dot(covi).dot(mu2) + np.log(pi2)
        dgamma = gamma2 - gamma1
        dbeta = beta1 - beta2
        b = dgamma/dbeta[1]
        m = - dbeta[0]/dbeta[1]
        y = m*x + b
        return y
LDA.demo()

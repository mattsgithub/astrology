import os
import numpy as np
import astrology


def get_updated_cov(cov, prev_mu, N, x):
    if N < 2:
        return cov
    xs = x - prev_mu
    XI = np.diag(xs)
    X = np.array([xs, xs])
    cov = (cov * (N - 2.0) + ((N - 1.0)/N) * XI.dot(X))/(N-1.0)
    return cov


def get_updated_mean(mu, N, x):
    """Update mean vector online

    Parameters
    ----------
        mu : numpy.array
            Mean vector
        x : numpy.array
            Data vector

    Returns
    -------
    mu : numpy.array
        Updated mean vector
    """
    c = 1.0 / N
    mu = c * ((N-1.0) * mu + x)
    return mu


def get_path(path):
    """
        Params
        ------
        path : str
            Seperated by period
    """
    path = path.split(".")
    ml_path = os.path.dirname(astrology.__file__)
    path_ = os.path.join(ml_path, *path)
    return path_

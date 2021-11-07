from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ebmpy.ssvm import MultivariateSSVM
from ebmpy.util import print_metrics

def run_demo():

    # The loss we want to use (here, I use f1-score)
    def loss_fn(tn, fp, fn, tp):
        return tp / (tp + .5 *(fp + fn))

    # Break into training / test
    df_train = pd.read_csv('./Datasets/train.csv')
    df_test = pd.read_csv('./Datasets/test.csv')
    feature_cols = df_train.columns[1:]
    label_col = 'y'
    n_classes = 2
    n_features = len(feature_cols)
    n_examples = df_train.shape[0]
    print('\nSetup')
    print("*" * 15)
    print('loss=f1_score')
    print('n_train_examples = %s' % df_train.shape[0])
    print('n_test_examples = %s' % df_test.shape[0])
    print('n_classes = %s' % n_classes)
    print('n_features = %s' % n_features)

    # Select out right columns
    X_train, X_test = df_train[feature_cols].values, df_test[feature_cols].values
    y_train, y_test = df_train[label_col].values, df_test[label_col].values

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    # Let's normalize our data
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Handy for later
    n_train_examples = X_train.shape[0]
    n_test_examples = X_test.shape[0]

    print('\nRandom')
    print("*" * 15)
    n = len(y_test.ravel())

    losses = []
    for _ in range(1000):
        y_pred = np.random.choice([-1, 1], n)
        tn, fp, fn, tp = confusion_matrix(y_test.ravel(), y_pred).ravel()
        losses.append(loss_fn(tn, fp, fn, tp))
    lower = np.percentile(losses, 5)
    mid = np.percentile(losses, 50)
    upper = np.percentile(losses, 95)
    print('test_loss_mid=%s' % lower)
    print('test_loss_lower=%s' % mid)
    print('test_loss_upper=%s' % upper)

    print('\nSVM')
    print("*" * 15)
    from sklearn import svm
    C = 0.01
    clf = svm.SVC(C = C)
    print(f'Fitting SVM. C={C}')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test.ravel(), y_pred).ravel()
    print(f'test_loss={loss_fn(tn, fp, fn, tp)}')

    print('\nSVM Perm Algorithm')
    print("*" * 15)
    energy_model = MultivariateSSVM(C=C, loss_fn=loss_fn)
    energy_model.fit(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    run_demo()
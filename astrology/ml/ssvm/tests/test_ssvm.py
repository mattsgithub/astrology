from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pystruct.learners import NSlackSSVM

#from ebmpy.ssvm import get_all_possible_confusion_matrices
#from ebmpy.ssvm import get_y_min
#from ebmpy.ssvm import MultivariateSSVM
from ebmpy.ssvm_pystruct import PystructSSVM
from ebmpy.dataset import get_water_quality_dataset
#from ebmpy.model import AutoEncoder
from ebmpy.util import print_metrics



def get_center_dataset(n_classes, n_features, n_examples):
    
    label_col = 'y'
    
    feature_cols = ['f%s' % i for i in range(1, n_features + 1)]
    X, y = sklearn.datasets.make_blobs(n_samples=n_examples,
                                       n_features=n_features, centers=n_classes,
                                       cluster_std=6.0,
                                       center_box=(-10.0, 10.0),
                                       shuffle=True, random_state=None)
    df = pd.DataFrame(X, columns=feature_cols)
    y[y == 0] = -1
    df[label_col] = y
    
    return df, feature_cols, label_col


def test_multivariate_ssvm():

    def loss_fn(tn, fp, fn, tp):
        return tp / (tp + .5 *(fp + fn))

    #def loss_fn(tn, fp, fn, tp):
    #    return (fp + fn * 1.) / (tn + fp + fn + tp)

    n_classes = 2
    n_features = 3
    n_examples = 600

    df, feature_cols, label_col = get_center_dataset(n_classes=n_classes,
                                                     n_features=n_features,
                                                     n_examples=n_examples)

    # Break into training / test
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=22)
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

    # Let's normalize our data
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Handy for later
    n_train_examples = X_train.shape[0]
    n_test_examples = X_test.shape[0]

    print('\nRandom Algorithm')
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


    # Pystruct SSVM
    print('\n Pystruct SSM Algorithm')
    print('*' * 15)
    model = PystructSSVM(X_train, y_train, X_test, y_test, loss_fn)
    clf = NSlackSSVM(model, C=1.0, batch_size=-1)
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test.ravel(), y_pred).ravel()
    print('test_loss={}' % loss_fn(tn, fp, fn, tp))
    
    # Simple SVM
    print('\nSVM Algorithm')
    print("*" * 15)
    from sklearn import svm
    C = 1.
    clf = svm.SVC(C = C)
    print('Fitting SVM. C={}' % C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test.ravel(), y_pred).ravel()
    print('test_loss={}' % loss_fn(tn, fp, fn, tp))

    """
    print('\nSSVM Algorithm')
    print("*" * 15)
    energy_model = MultivariateSSVM(loss_fn=loss_fn, eps=0.01)
    energy_model.fit(X_train, y_train, X_test, y_test)
    """


def test_get_all_possible_confusion_matrices():
    
    y_true = np.array([1, 0, 1, 0, 1, 1, 1, 0])
    
    # All possible y outcomes
    y_preds = np.array(list(product([0, 1], repeat=y_true.shape[0])))

    degenerate_cases = defaultdict(int)
    
    for y_pred in y_preds:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        degenerate_cases[tn, fp, fn, tp] += 1

    print("Number of possible y outcomes: {}" % len(y_preds))
    print("Number of degenerate cases: {}" % len(degenerate_cases))
    
    confusion_matrices = get_all_possible_confusion_matrices(y_true)
    print("Number of confusion metrics: {}" % len(confusion_matrices))
    
    assert set(degenerate_cases.keys()) == set(confusion_matrices)


def test_get_y_min():
    np.random.seed(22)
    
    n_features = 10
    n_examples = 6
        
    y_true = np.random.choice((-1, 1), n_examples)
    w = np.random.random(n_features)
    X = np.random.random((n_examples, n_features))

    # Computed once
    dot_prod = X.dot(w)

    # Find all y_preds that map to a given confusion matrix
    y_preds = np.array(list(product([-1, 1], repeat=y_true.shape[0])))
    mapping = defaultdict(list)
    for y_pred in y_preds:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        mapping[tn, fp, fn, tp].append(y_pred)

    new_mapping = dict()
    for table, y_preds in mapping.items():
        tn, fp, fn, tp = table

        # Find y_pred that maps to lowest energy
        y_pred_min = None
        min_energy = float('inf')
        for y_pred in y_preds:
            energy = dot_prod.dot(y_pred)
            if energy < min_energy:
                min_energy = energy
                y_pred_min = y_pred

        new_mapping[table] = y_pred_min

    for table, y_pred in new_mapping.items():
        tn, fp, fn, tp = table
        y_min = get_y_min(y_true, dot_prod, tp, fp)
        assert np.array_equal(y_pred, y_min)


if __name__ == '__main__':
    #test_get_all_possible_confusion_matrices()
    #test_get_y_min()
    test_multivariate_ssvm()
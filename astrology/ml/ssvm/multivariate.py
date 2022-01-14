"""

Logging Xw over time might yield insights
There are 'N' components (one for each training example)
"""

from itertools import product
import logging

import numpy as np
import cvxpy as cp
from sklearn.metrics import confusion_matrix
import logging


class MultivariateSSVM:
    def __init__(self, loss_fn,
                 C=0.01, eps=0.1,
                 max_iterations=30):

        if C < 0.:
            raise ValueError(f'C={C} < 0. C must be greater than 0')

        self.C = C

        # TODO: How sensitive is eps to the problem?
        self.eps = eps
        self.loss_fn = loss_fn
        self.max_iterations = max_iterations

    @property
    def w(self):
        return self._w
  
    @w.setter
    def w(self, value):
        # I use an observer pattern. Since
        # when w updates, we need to update
        # quite a few other values
        self._w = value
        print(f'w={self._w}')
        print(f'||w||={np.linalg.norm(self._w)}')

        self.X_dot_w = self.X.dot(self._w)

        # By default, argsort sorts ascending
        # Flip to make it descending
        X_dot_w_y_pos_values = self.X_dot_w[self.I_y_pos]
        I = np.flip(np.argsort(X_dot_w_y_pos_values))
        self.I_Xw_largest_y_pos_indices = self.I_y_pos[I]

        X_dot_w_y_neg_values = self.X_dot_w[self.I_y_neg]
        I = np.argsort(X_dot_w_y_neg_values)
        self.I_Xw_smallest_y_neg_indices = self.I_y_neg[I]

        # Compute test loss
        if self.X_test is not None:
            y_pred = self.predict(self.X_test)
            tn, fp, fn, tp = confusion_matrix(self.y_test.ravel(), y_pred).ravel()
            print(f'test_loss={self.loss_fn(tn, fp, fn, tp)}')

    def predict(self, X):
        # What y makes
        # sum_i (w * xi) * yi
        # the smallest?
        #
        # The largest (w * xi) should get
        # a neg. The smallest should get a pos.
        # This is why after we take the sign
        # We flip it
        return -1 * np.sign(X.dot(self.w))

    @staticmethod
    def psi(X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return np.multiply(X, y).sum(axis=0)
    
    @staticmethod
    def get_all_possible_binary_confusion_matrices(y):
        
        N = y.shape[0]
        n_pos_examples = y[y == 1].shape[0]
        n_neg_examples = N - n_pos_examples

        confusion_matrices = []
        for fp in range(n_neg_examples + 1):
            tn = n_neg_examples - fp
            for tp in range(n_pos_examples + 1):
                fn = n_pos_examples - tp
                confusion_matrices.append((tn, fp, fn, tp))

        return confusion_matrices

    def get_y_most_violated_constraint(self):

        max_margin = float('-inf')
        for tn, fp, fn, tp in self.binary_confusion_matrices:
            loss = self.loss_fn(tn, fp, fn, tp)
            # We take all
            # (xi * w, y = 1) tuples
            #
            # We sort these by tuples
            # from largest to smallest (using first component)
            # We will modify the top 'fn' to be y = -1
            #
            # Clearly if xi * w > 0 we are generating the
            # largest negative values possible (our goal)
            # given this contraint
            #
            # Suppose xi * w < 0
            # Now we are making positive values, but at least
            # they are smallest positive values of all the
            # choices we have!
            #
            # This is the optimal strategy given the contraint
            y_ = self.y.copy()
            y_[self.I_Xw_largest_y_pos_indices[:fn]] = -1
            y_[self.I_Xw_smallest_y_neg_indices[:fp]] = 1

            energy = self.psi(self.X, y_).dot(self.w)

            # Note the minus energy
            # In "A Support Vector Method for Multivariate Performance Measures"
            # This would be loss + energy
            # But here I treat this algorithm as an energy based models
            margin = loss - energy

            # If ties, first one is picked
            # TODO: What are the mathematical implications of this?
            # What type of bias do we introduce (if any)?
            if margin > max_margin:
                max_margin = margin
                y_max = y_

        return y_max

    def fit(self, X, y, X_test=None, y_test=None):
        print('Fitting SSVM')
        print(f'C={self.C}')
        print(f'eps={self.eps}')

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.y_constraints = None
        self.delta_constraints = None
        self.loss_constraints = None
        self.slack = None

        self.n_training_examples = X.shape[0]

        # Compute once and store.
        self.psi_X_y = self.psi(X, y)
        self.I_y_pos = np.where(self.y == 1)[0]
        self.I_y_neg = np.where(self.y == -1)[0]

        # The vector we are trying to learn from data
        self.w = np.random.random(X.shape[1])

        self.binary_confusion_matrices = self.get_all_possible_binary_confusion_matrices(y)

        # We will stop if max is reached
        # OR
        # constraints stop being added 
        for i in range(self.max_iterations):

            y_ = self.get_y_most_violated_constraint()

            # Already calculated this before, but it
            # makes code less complex (easier to read)
            # to just calculate again here
            psi_X_y_ = self.psi(X, y_)

            delta_X_y = psi_X_y_ - self.psi_X_y

            tn, fp, fn, tp = confusion_matrix(self.y, y_).ravel()
            loss = self.loss_fn(tn, fp, fn, tp)

            if self.loss_constraints is None:
                slack = 0
            else:
                energies = self.loss_constraints - self.delta_constraints.dot(self.w)
                max_energy = np.max(energies)
                slack = max(0, max_energy)

            n_constraints = 0 if self.y_constraints is None else len(self.y_constraints)

            # Check if we already have this constraint
            # TODO: Is there a better way to do this check?
            new_constraint = True
            if self.y_constraints is not None:
                for yi in self.y_constraints:
                    if np.array_equal(y_, yi):
                        new_constraint = False
                        break

            if ((loss - self.w.dot(delta_X_y)) > slack + self.eps) and new_constraint:
                # Is this the first time adding a constraint?
                if n_constraints == 0:
                    self.y_constraints = np.array([y_])
                    self.delta_constraints = np.array([delta_X_y])
                    self.loss_constraints = np.array([loss])
                else:
                    self.y_constraints = np.vstack((self.y_constraints, y_))
                    self.delta_constraints = np.vstack((self.delta_constraints, delta_X_y))
                    self.loss_constraints = np.append(self.loss_constraints, loss)
                print(f'\nn_contraints={len(self.y_constraints)}')

                # Declare variable here
                # so we can reference it in
                # our constraints
                w = cp.Variable(X.shape[1])
                slack = cp.Variable()

                # Create constraints for solver
                constraints = [slack >= 0]
                for y_ in self.y_constraints:

                    # Compute delta
                    psi_X_y_ = self.psi(X, y_)
                    delta_X_y = psi_X_y_ - self.psi_X_y
                  
                    tn, fp, fn, tp = confusion_matrix(y, y_).ravel()
                    loss = self.loss_fn(tn, fp, fn, tp)

                    # TODO: How can I QA these constraints?
                    # Can only use '@' in Python 3.5 >=
                    constraints.append(w @ delta_X_y >= loss - slack)

                # Set objective function
                objective = cp.Minimize(0.5 * cp.power(cp.norm(w), 2) + self.C * slack)

                # Setup the problem    
                prob = cp.Problem(objective, constraints)
                prob.solve()
            
                if prob.status != 'optimal':
                    print(f'Problem could not be solved. Solution is {prob.status}')
                    print('Will use last w value. Quitting')
                    break
                
                self.w = w.value
                self.slack = slack.value
                print(f'slack={self.slack}')

            if n_constraints == len(self.y_constraints):
                print('No new constraints added. Quiting.')
                break
from os import X_OK
import numpy as np
from sklearn.metrics import confusion_matrix
from pystruct.models.base import StructuredModel


class PystructSSVM(StructuredModel):
    def __init__(self, X, y, X_test, y_test, loss_fn):
        # Compute once and store.
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.psi_X_y = self.psi(self.X, self.y)
        self.I_y_pos = np.where(self.y == 1)[0]
        self.I_y_neg = np.where(self.y == -1)[0]
        self.loss_fn = loss_fn
        self.binary_confusion_matrices = self.get_all_possible_binary_confusion_matrices(y)
        self.size_joint_feature = self.X.shape[1]
        self.set_w(np.random.random(self.X.shape[1]))

    def predict(self, X):
        # What y makes
        # sum_i (w * xi) * yi
        # the smallest?
        #
        # The largest (w * xi) should get
        # a neg. The smallest should get a pos.
        # This is why after we take the sign
        # We flip it
        return -np.sign(X.dot(self._w))

    @staticmethod
    def psi(X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return np.multiply(X, y).sum(axis=0)

    def set_w(self, value):
        # I use an observer pattern. Since
        # when w updates, we need to update
        # quite a few other values
        self._w = value
        print('w=%s' % self._w)

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
        """
        if self.X_test is not None:
            y_pred = self.predict(self.X_test)
            import pdb; pdb.set_trace()
            tn, fp, fn, tp = confusion_matrix(self.y_test.ravel(), y_pred).ravel()
            print('test_loss={}' % self.loss_fn(tn, fp, fn, tp))
        """

    def joint_feature(self, x, y):
        return self.psi(x, y)

    def loss(self, y, y_hat):
        tn, fp, fn, tp = confusion_matrix(y.ravel(), y_hat).ravel()
        return tp / (tp + .5 *(fp + fn))

    def inference(self, x, w, relaxed=None, constraints=None):
        return -1 * np.sign(x.dot(w))

    def loss_augmented_inference(self, x, y, w, relaxed=None):
        self.set_w(w)
        y_hat = self.get_y_most_violated_constraint()
        import pdb; pdb.set_trace()
        return y_hat

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

            energy = self.psi(self.X, y_).dot(self._w)

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
import numpy as np


class Word2Vec(object):
    def __init__(self, m=100, alpha=0.01):
        self.W1_T = None
        self.W2_T = None
        self.m = m
        self.alpha = alpha
        self.words = None
        self.V = None

    @staticmethod
    def softmax(u):
        """np.exp(710) raises overflow problem
           subtracting max helps with this
        """
        max_ = u.max()
        u_ = u - max_
        y_ = np.exp(u_)
        sum_ = np.sum(y_)
        y = y_ / sum_
        return y

    @staticmethod
    def forward_propagate(W1_T,
                          W2_T,
                          x):
        h = W1_T.dot(x)
        u = W2_T.dot(h)
        y = Word2Vec.softmax(u)
        return h, u, y

    @staticmethod
    def get_one_hot_vector(index, N):
        x = np.zeros(N)
        x[index] = 1.
        return x

    def get_word_as_vector(self, w):
        index = self.words[w]
        return self.W1_T[:, index]

    def _add_word_to_vocab(self, w):
        if w not in self.words:
            self.words[w] = self.index
            self.index += 1

    def train(self, words):
        # words are assumed
        # to be in a sequence
        self.words = dict()
        self.index = 0
        X = []
        T = []

        target = words[0]
        self._add_word_to_vocab(target)
        # Step through each word
        for j in xrange(1, len(words)):
            w = words[j]
            self._add_word_to_vocab(w)
            X.append(w)
            T.append(target)
            target = w

        # Convert to vectors since we know size
        # now all possible words in vocabulary
        self.V = len(self.words)
        for i in xrange(len(X)):
            X[i] = Word2Vec.get_one_hot_vector(self.words[X[i]], self.V)
            T[i] = Word2Vec.get_one_hot_vector(self.words[T[i]], self.V)

        # Initialize to small weights
        self.W1_T = np.random.normal(loc=0.0, scale=0.01, size=(self.m, self.V))
        self.W2_T = np.random.normal(loc=0.0, scale=0.01, size=(self.V, self.m))

        # Update for each training example
        for x, t in zip(X, T):
            h, u, y = Word2Vec.forward_propagate(self.W1_T,
                                                 self.W2_T,
                                                 x)

            e = y - t
            # Update first matrix
            for i in xrange(self.m):
                for j in xrange(self.V):
                    self.W2_T[j, i] = self.W2_T[j, i] - self.alpha * e[j] * h[i]

            # Update second matrix
            for i in xrange(self.m):
                for j in xrange(self.V):
                    sum_ = sum([e[j] * self.W2_T[j, i] for j in xrange(self.V)])
                    self.W1_T[i, j] = self.W1_T[i, j] - self.alpha * x[j] * sum_

if __name__ == '__main__':
    words = []
    with open('/Users/MattJohnson/Data/books/tom_sawyer.txt') as f:
        for line in f:
            words.extend(line.split(' '))
            if len(words) > 100:
                break

    wv = Word2Vec()
    wv.train(words)
    import pdb; pdb.set_trace()

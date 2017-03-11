"""Builds a Bernoulli naive Bayes classifier
"""

from math import log
import glob
from collections import Counter


def get_features(text):
    """Extracts features from text

    Args:
        text (str): A blob of unstructured text
    """
    return set([w.lower() for w in text.split(" ")])


class BernoulliNBTextClassifier(object):

    def __init__(self):
        self._log_priors = None
        self._cond_probs = None
        self.features = None

    def train(self, documents, labels):
        """Train a Bernoulli naive Bayes classifier

        Args:
            documents (list): Each element in this list
                is a blog of text
            labels (list): The ground truth label for
                each document
        """

        """Compute log( P(Y) )
        """
        label_counts = Counter(labels)
        N = float(sum(label_counts.values()))
        self._log_priors = {k: log(v/N) for k, v in label_counts.iteritems()}

        """Feature extraction
        """
        # Extract features from each document
        X = [set(get_features(d)) for d in documents]

        # Get all features
        self.features = set([f for features in X for f in features])

        """Compute log( P(X|Y) )

           Use Laplace smoothing
           n1 + 1 / (n1 + n2 + 2)
        """
        self._cond_probs = {l: {f: 0. for f in self.features} for l in self._log_priors}

        # Step through each document
        for x, l in zip(X, labels):
            for f in x:
                self._cond_probs[l][f] += 1.

        # Now, compute log probs
        for l in self._cond_probs:
            N = label_counts[l]
            self._cond_probs[l] = {f: (v + 1.) / (N + 2.) for f, v in self._cond_probs[l].iteritems()}

    def predict(self, text):
        """Make a prediction from text
        """

        # Extract features
        x = get_features(text)

        pred_class = None
        max_ = float("-inf")

        # Perform MAP estimation
        for l in self._log_priors:
            log_sum = self._log_priors[l]
            for f in self.features:
                prob = self._cond_probs[l][f]
                log_sum += log(prob if f in x else 1. - prob)
            if log_sum > max_:
                max_ = log_sum
                pred_class = l

        return pred_class


def get_labeled_data(type_):
    """Get data from:
        http://openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex6materials/ex6DataEmails.zip
        Create a folder named 'emails' with content from ex6DataEmails in
        same directory as this script
    """
    examples = []
    labels = []

    file_names = glob.glob('./emails/spam-{0}/*.txt'.format(type_))
    for n in file_names:
        f = open(n)
        examples.append(f.read())
        labels.append('spam')
        f.close()

    file_names = glob.glob('./emails/nonspam-{0}/*.txt'.format(type_))
    for n in file_names:
        f = open(n)
        examples.append(f.read())
        labels.append('nonspam')
        f.close()

    return examples, labels

if __name__ == "__main__":
    train_docs, train_labels = get_labeled_data('train')
    test_docs, test_labels = get_labeled_data('test')

    # Train model
    print('Number of training examples: {0}'.format(len(train_labels)))
    print('Number of test examples: {0}'.format(len(test_labels)))
    print('Training model...')
    nb = BernoulliNBTextClassifier()
    nb.train(train_docs, train_labels)
    print('Training complete!')
    print('Number of features found: {0}'.format(len(nb.features)))


    # Simple error test metric
    print('Testing model...')
    f = lambda doc, l: 1. if nb.predict(doc) != l else 0.
    num_missed = sum([f(doc, l) for doc, l in zip(test_docs, test_labels)])

    N = len(test_labels) * 1.
    error_rate = round(100. * (num_missed / N), 3)

    print('Error rate of {0}% ({1}/{2})'.format(error_rate, int(num_missed), int(N)))

from random import random
from math import log

from astrology.corpora import TwentyNewsGroupCorpus
from astrology.metric import cross_validate



class BernoulliNB(object):
    
    def __init__(self):
        self.reset()
     
    def _get_features(self, text):
        """Get features as a set for this training example"""
        return set([w.lower() for w in text.split(" ")])

    def reset(self):
        self._N = 0.0
        self._features = set()
        self._class_counts = dict()
        self._class_feature_counts = dict()
    
    def observe(self, text, class_):
        
        # Update observation count
        self._N += 1.0
        
        features = self._get_features(text)
        
        [self._features.add(f) for f in features]
        
        # Update count dicts if this is the first
        # time observing this class
        if class_ not in self._class_counts:
            self._class_counts[class_] = 0.0
            self._class_feature_counts[class_] = dict()
               
        # Update class count
        self._class_counts[class_] += 1.0
        
        # Update feature count
        for feature in features:
            if feature not in self._class_feature_counts[class_]:
                self._class_feature_counts[class_][feature] = 0.0
            self._class_feature_counts[class_][feature] += 1.0

    def predict(self, text):
        
        if (self._N) == 0.0:
            raise Exception("Need to observe at least one example")
        
        features = self._get_features(text)
        
        pred_class = None
        max_ = float("-inf")
        
        for class_ in self._class_counts:
            
            # Number of training examples with this class
            M = self._class_counts[class_]
            
            log_sum = log(M/self._N)
            for f in features:
                # Laplace smoothing
                # Add one if feature doesn't exist
                a = self._class_feature_counts[class_].get(f,0.0) + 1.0     
                b = (M + len(self._features))*1.0
                log_sum += log(a/b)
            if log_sum >= max_:
                max_ = log_sum
                pred_class = class_
        return pred_class


class MultinomialNB(object):
    
    def __init__(self):
        
        # Stores features counts for a given class
        self._class_feature_counts = dict()
        
        # Stores class counts
        self._class_counts = dict()
        
        # Number of observations
        self._N = 0.0
        
        # All features
        self._features = set()
     
    def _get_features(self, text):
        """Get features as a set for this training example"""
        return set([w.lower() for w in text.split(" ")])
    
    def observe(self, text, class_):
        
        # Update observation count
        self._N += 1.0
        
        features = self._get_features(text)
        
        [self._features.add(f) for f in features]
        
        # Update count dicts if this is the first
        # time observing this class
        if class_ not in self._class_counts:
            self._class_counts[class_] = 0.0
            self._class_feature_counts[class_] = dict()
               
        # Update class count
        self._class_counts[class_] += 1.0
        
        # Update feature count
        for feature in features:
            if feature not in self._class_feature_counts[class_]:
                self._class_feature_counts[class_][feature] = 0.0
            self._class_feature_counts[class_][feature] += 1.0

    def predict(self, text):
        
        if (self._N) == 0.0:
            raise Exception("Need to observe at least one example")
        
        features = self._get_features(text)
        
        pred_class = None
        max_ = float("-inf")
        
        for class_ in self._class_counts:
            
            # Number of training examples with this class
            M = self._class_counts[class_]
            
            log_sum = log(M/self._N)
            for f in features:
                # Laplace smoothing
                # Add one if feature doesn't exist
                a = self._class_feature_counts[class_].get(f,0.0) + 1.0     
                b = (M + len(self._features))*1.0
                log_sum += log(a/b)
            if log_sum >= max_:
                max_ = log_sum
                pred_class = class_
        return pred_class

def test_bernouli_naive_bayes():
    bnb = BernoulliNB()
    corpus = TwentyNewsGroupCorpus()
    cross_validate(corpus, bnb)

if __name__ == "__main__":
    test_bernouli_naive_bayes()

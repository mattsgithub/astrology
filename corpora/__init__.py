from random import shuffle

from collections import namedtuple
from pymongo import MongoClient


class TwentyNewsGroupCorpus(object):
    def __init__(self):
        mc = MongoClient()
        db = mc.astrology
        
        Example = namedtuple('Example', ['text', 'label'])

        # Get collections
        coll = db.corpora.twenty_news_group
        meta_coll = db.corpora.twenty_news_group.meta

        # Set meta info
        meta = meta_coll.find_one()
        self._labels = set(meta.get("labels"))

        self._docs_by_label = {l:[] for l in self._labels}
        self._all_training_examples = [] 
        self._test_examples = []
        self._training_examples = []

        for doc in coll.find():
            label = doc.get("label")
            text = doc.get("text")
            tag = doc.get("tag")
            example = Example(text=text, label=label)

            if tag == "test":
                self._test_examples.append(example)
            else:
                self._training_examples.append(example)

        mc.close()

        self._all_training_examples = self._training_examples + self._test_examples

        shuffle(self._all_training_examples)

    def get_labels(self):
        return self._labels

    def get_all_training_examples(self):
        return self._training_examples + self._test_examples
    
    def get_training_examples(self):
        return self._training_examples 

    def get_test_examples(self):
        return self._test_examples

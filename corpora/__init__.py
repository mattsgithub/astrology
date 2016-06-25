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
        self._training_examples = [] 

        for doc in coll.find():
            label = doc.get("label")
            text = doc.get("text")
            example = Example(text=text, label=label)
            self._training_examples.append(example)

        shuffle(self._training_examples)

        # Close connection
        mc.close()

    def get_labels(self):
        return self._labels

    def get_training_examples(self):
        return self._training_examples

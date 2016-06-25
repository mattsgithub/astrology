import os

import bson
from pymongo import MongoClient

from astrology.util import get_path


def load_20_news_group():
    """ Loads the 20 news group corpus into
        a mongo database
    """
    mc = MongoClient()
    db = mc["astrology"]

    coll_name = "corpora.twenty_news_group"
    meta_coll_name = "corpora.twenty_news_group.meta"

    # Drop if already exists
    db.drop_collection(coll_name)
    db.drop_collection(meta_coll_name)

    coll = db[coll_name]
    meta_coll = db[meta_coll_name]

    labels = set()

    for batch in get_20_news_group(300, labels):
        coll.insert_many(batch)

    meta_doc = {"labels": list(labels)}
    meta_coll.insert_one(meta_doc)

    coll.create_index("label")

    mc.close()


def get_20_news_group(batch_size, labels):
    """ Yields a batch of documents read from disk

        Params
        ------
        batch_size : int
            How many documents to yield at a time

        labels : str
            A reference to an empty set. Fills up the
            set as labels are discovered
    """

    docs = []

    test_path = get_path("corpora.data.20newsgroup.20news-bydate-test")
    train_path = get_path("corpora.data.20newsgroup.20news-bydate-train")

    test_dir_names = os.listdir(test_path)
    train_dir_names = os.listdir(train_path)

    dir_names = test_dir_names + train_dir_names

    for i, dir_name in enumerate(dir_names):
        label = dir_name
        dir_path = os.path.join(test_path, dir_name)
        file_names = os.listdir(dir_path)

        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)

            f = open(file_path, "r")
            text = f.read()
            f.close()

            # Because I'm too lazy to figure out
            # fundamentally what's going on
            # right now. So I ignore errors :)
            text = unicode(text, errors="ignore")

            labels.add(label)
            docs.append({"text": text, "label": label})

            if len(docs) > batch_size:
                yield docs
                docs = []

    # Any leftovers?
    if len(docs) > 0:
        yield docs

load_20_news_group()

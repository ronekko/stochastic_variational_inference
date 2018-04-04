# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:38:04 2018

@author: sakurai
"""

import gzip
from pathlib import Path
import urllib

import h5py
import numpy as np


def _download_uci_corpus(dataset_name, dataset_location='~/dataset/bow'):
    url_base = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                'bag-of-words/')
    bow_url = url_base + 'docword.{}.txt.gz'.format(dataset_name)
    vocab_url = url_base + 'vocab.{}.txt'.format(dataset_name)
    bow_path = Path(dataset_location, 'docword.{}.txt.gz'.format(dataset_name))
    vocab_path = Path(dataset_location, 'vocab.{}.txt'.format(dataset_name))

    Path(dataset_location).mkdir(exist_ok=True)
    if not bow_path.exists():
        urllib.request.urlretrieve(bow_url, bow_path)
    if not vocab_path.exists():
        urllib.request.urlretrieve(vocab_url, vocab_path)


def _create_hdf5(dataset_name, dataset_location='~/dataset/bow'):
    if dataset_name not in ('kos', 'enron', 'nips', 'nytimes', 'pubmed'):
        raise ValueError("`dataset_name` must be 'kos', 'enron', 'nips'"
                         ", 'nytimes' or 'pubmed'")

    dataset_location = Path(dataset_location).expanduser()
    docwords_filepath = dataset_location / 'docword.{}.txt.gz'.format(
        dataset_name)
    vocab_filepath = dataset_location / 'vocab.{}.txt'.format(dataset_name)

    hdf5_filepath = Path(dataset_location, '{}.hdf5'.format(dataset_name))
    if hdf5_filepath.exists():
        return

    print(hdf5_filepath)
    f = h5py.File(hdf5_filepath)

    # Create bag-of-words (pairs of an array of words and an array of counts)
    docwords_file = gzip.open(docwords_filepath)
    n_documents = int(docwords_file.readline())
    docwords_file.readline()  # Read and discard a line (n_words)
    docwords_file.readline()  # Read and discard a line (n_nonzero)

    dt = h5py.special_dtype(vlen=np.int32)
    ds = f.create_dataset('bow', (n_documents, 2), dtype=dt)

    current_document = 0
    words = []
    counts = []
    for line in docwords_file:
        d, w, n = [int(num) for num in line.decode('utf-8').split(' ')]
        if d - 1 != current_document:
            words = np.array(words, np.int32) - 1  # Adjust word index 0-origin
            counts = np.array(counts, np.int32)
            ds[current_document] = np.vstack((words, counts))
            current_document = d - 1
            words = []
            counts = []

        words.append(w)
        counts.append(n)

    words = np.array(words, np.int32) - 1  # Adjust word index 0-origin
    counts = np.array(counts, np.int32)
    ds[current_document] = np.vstack((words, counts)).astype(np.int32)
    docwords_file.close()

    # Create vocabulary
    with open(vocab_filepath) as vocab_file:
        words = [word.strip() for word in vocab_file.readlines()]
    n_vocab = len(words)
    dt = h5py.special_dtype(vlen=str)
    ds = f.create_dataset('vocab', (n_vocab,), dtype=dt)
    ds[:] = words

    f.flush()
    f.close()


class _DatasetBase(object):
    dataset_name = None

    def __init__(self, dataset_location='~/dataset/bow'):
        if self.dataset_name is None:
            raise NotImplementedError

        dataset_name = self.dataset_name
        _download_uci_corpus(dataset_name, dataset_location)
        _create_hdf5(dataset_name, dataset_location)
        hdf5_filepath = Path(dataset_location, '{}.hdf5'.format(dataset_name))
        self._hdf5 = h5py.File(hdf5_filepath, 'r')
        self.docs = self._hdf5['bow']
        self.vocabulary = np.array(self._hdf5['vocab'])
        self.num_docs = len(self.docs)
        self.num_terms = len(self.vocabulary)

        self.id2word = self.vocabulary

    def __del__(self):
        self._hdf5.close()


class KosDataset(_DatasetBase):
    dataset_name = 'kos'


class NipsDataset(_DatasetBase):
    dataset_name = 'nips'


class EnronDataset(_DatasetBase):
    dataset_name = 'enron'


class NytimesDataset(_DatasetBase):
    dataset_name = 'nytimes'


class PubmedDataset(_DatasetBase):
    dataset_name = 'pubmed'


if __name__ == '__main__':
    dataset_location = r'E:\Dataset\bow'

    dataset = KosDataset(dataset_location)
    dataset = NipsDataset(dataset_location)
    dataset = EnronDataset(dataset_location)
    dataset = NytimesDataset(dataset_location)

#!/usr/bin/env python

import click

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from hmmlearn import hmm

import codecs

from constants import *

import pickle

@click.group()
def baseline():
    """Performs baseline system experiments."""
    pass

#@click.option('--n-components', type=int, required=True, help='Number of hidden states')
@click.command()
def multinomial_hmm():
    "Trains a discrete emission HMM. Uses ./scratch/concat_corpus.txt"
    np.random.seed(42)

    corpus = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())[:1000]
    char_to_idx = { c:i for i,c in enumerate(set(corpus)) }

    ohe = OneHotEncoder(dtype=np.int, sparse=False)
    corpus_enc = ohe.fit_transform([[char_to_idx[c]] for c in corpus])

    model = hmm.MultinomialHMM(n_components=len(char_to_idx), verbose=True)
    model.fit(corpus_enc)

    pickle.dump(model, open('hmm.model', 'wb'))


map(baseline.add_command, [
    multinomial_hmm
])

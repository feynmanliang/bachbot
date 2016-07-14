#!/usr/bin/env python

import click
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from hmmlearn import hmm
from rnnrbm.rnnrbm import RnnRbm

import codecs

from music21 import *

from chorales import standardize_key
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
    # TODO: This is too slow currently even with only 1000 samples
    np.random.seed(42)

    corpus = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())[:1000]
    char_to_idx = { c:i for i,c in enumerate(set(corpus)) }

    ohe = OneHotEncoder(dtype=np.int, sparse=False)
    corpus_enc = ohe.fit_transform([[char_to_idx[c]] for c in corpus])

    model = hmm.MultinomialHMM(n_components=len(char_to_idx), verbose=True)
    model.fit(corpus_enc)

    pickle.dump(model, open('hmm.model', 'wb'))

@click.command()
def rnnrbm():
    ## Prepare MIDI scores
    # files = []
    # for score in corpus.chorales.Iterator(
    #         numberingSystem='bwv',
    #         returnType='stream'):
    #     # only consider 4/4
    #     if not score.getTimeSignatures()[0].ratioString == '4/4': continue

    #     bwv_id = score.metadata.title
    #     print('Processing BWV {0}'.format(bwv_id))

    #     score = standardize_key(score)
    #     key = score.analyze('key')

    #     mf = midi.translate.streamToMidiFile(score)
    #     fp = SCRATCH_DIR + '/{0}-{1}-chord-constant-t.mid'.format(bwv_id, key.mode)
    #     mf.open(fp, 'wb')
    #     mf.write()
    #     mf.close()
    #     files.append(fp)
    files = glob.glob(SCRATCH_DIR + "/*.mid")
    model = RnnRbm()
    model.train(files, batch_size=100, num_epochs=200)
    model.generate('sample0.mid')
    model.generate('sample1.mid')
    model.generate('sample2.mid')
    model.generate('sample3.mid')
    model.generate('sample4.mid')
    pylab.show()

map(baseline.add_command, [
    multinomial_hmm,
    rnnrbm
])

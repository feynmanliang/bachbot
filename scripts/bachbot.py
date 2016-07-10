import click

from music21 import environment

from chorales import chorales
from corpus_utils import concatenate_corpus
from keras_net import keras
from score import score
from torch_rnn import make_h5, train, sample, postprocess_utf, postprocess_utf_constant_timestep
from decode import decode

@click.group()
def cli():
    us = environment.UserSettings()
    # ~/.music21rc should be edited to support using musescore for score.show()
    # us.create()
    pass

# instantiate the CLI
map(cli.add_command, [
    chorales,
    concatenate_corpus,
    make_h5,
    train,
    sample,
    postprocess_utf,
    postprocess_utf_constant_timestep,
    keras,
    score,
    decode,
])

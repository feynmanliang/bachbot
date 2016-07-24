import click

from music21 import environment

from datasets import datasets
from score import score
from torch_rnn import make_h5, train, sample, postprocess_utf, postprocess_utf_constant_timestep
from decode import decode
from baseline import baseline
from analysis import analysis

@click.group()
def cli():
    us = environment.UserSettings()
    # ~/.music21rc should be edited to support using musescore for score.show()
    # us.create()
    pass

# instantiate the CLI
map(cli.add_command, [
    datasets,
    make_h5,
    train,
    sample,
    postprocess_utf,
    postprocess_utf_constant_timestep,
    score,
    decode,
    baseline,
    analysis
])

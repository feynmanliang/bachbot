import click

from music21 import environment

from bach_chorales import prepare_bach_chorales_mono
from concatenate_corpus import concatenate_corpus
from torch_rnn import make_h5, train, sample, postprocess_utf

@click.group()
def cli():
    us = environment.UserSettings()
    # ~/.music21rc should be edited to support using musescore for score.show()
    # us.create()
    pass

# instantiate the CLI
map(cli.add_command, [
    prepare_bach_chorales_mono,
    concatenate_corpus,
    make_h5,
    train,
    sample,
    postprocess_utf
])

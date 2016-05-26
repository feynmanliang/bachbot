import click

from music21 import environment

from concatenate_corpus import concatenate_corpus
from kerntools.extract_melody import extract_melody
from kerntools.renumber_measures import renumber_measures
from kerntools.scrape_humdrum import scrape_humdrum

@click.group()
def cli():
    us = environment.UserSettings()
    # ~/.music21rc should be edited to support using musescore for score.show()
    # us.create()
    pass

# kern.humdrum.org and kern file processing
cli.add_command(scrape_humdrum)
cli.add_command(extract_melody)

# torch-rnn
cli.add_command(concatenate_corpus)

# kern file post-processing
cli.add_command(renumber_measures)

import click

from music21 import environment

from bach_chorales import prepare_bach_chorales_mono

@click.group()
def cli():
    us = environment.UserSettings()
    # ~/.music21rc should be edited to support using musescore for score.show()
    # us.create()
    pass

# input preprocessing
cli.add_command(prepare_bach_chorales_mono)

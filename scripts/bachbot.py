import click

from kerntools.scrape_humdrum import scrape_humdrum
from kerntools.concatenate_corpus import concatenate_corpus
from kerntools.renumber_measures import renumber_measures

from constants import BACHBOT_DIR

@click.group()
def cli():
    pass

import glob
from music21 import *
@click.command()
@click.option('--file-list', type=click.File('rb'),
        help='A text file where each line is a path to an input kern file.')
@click.option('--out-dir', default=BACHBOT_DIR + '/scratch', type=click.Path(exists=True),
        help='The directory to write the *-mono.kern files to.')
def extract_melody(file_list, out_dir):
    """Extracts a monophonic melody voices from a list of kern file.

    We currently define the melody line to be the soprano voice.
    """

    if not file_list:
        file_list = glob.glob('{0}/corpus/Bach+Johann/chor*.krn'.format(BACHBOT_DIR))
    else:
        file_list = file_list.readlines()

    for f in file_list[0:2]:
        print f
        score = converter.parseFile(f)
        print score.analyze('key')
        #print score.show('text')
        partStream = score.parts.stream()
        for p in partStream:
            #print p[0][1]
            continue

cli.add_command(scrape_humdrum)
cli.add_command(extract_melody)
cli.add_command(concatenate_corpus)
cli.add_command(renumber_measures)

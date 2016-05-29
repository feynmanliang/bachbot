import click

from constants import *

@click.command()
@click.argument('files', nargs=-1, required=True)
@click.option('-o', '--output', type=click.File('w+'), default=SCRATCH_DIR + '/concat_corpus.txt')
def concatenate_corpus(files, output):
    """Concatenates individual files together into single corpus.

    Try `bachbot concatenate_corpus scratch/*.utf`.
    """
    for fp in files:
        with open(fp, 'rb') as fd:
            output.write(START_DELIM + '\n' + fd.read() + '\n' + END_DELIM + '\n')



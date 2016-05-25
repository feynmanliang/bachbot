import click
import glob
import os

from constants import BACHBOT_DIR

@click.command()
@click.option('--file-list', type=click.File('rb'),
        help='A text file where each line is a path to an input kern file.')
@click.option('--out-dir', default=BACHBOT_DIR + '/scratch', type=click.Path(exists=True),
        help='The directory to write the processed and concatenated corpus (with same filename as the file-list).')
def concatenate_corpus(file_list, out_dir):
    """Preprocesses raw kern files and concatenates into corpus."""
    if not file_list:
        file_list = glob.glob('{0}/scratch/*-mono.xml'.format(BACHBOT_DIR))
        out_filepath = out_dir + '/concat.txt'
    else:
        file_list = file_list.readlines()
        out_filepath = out_dir + '/' + os.path.basename(file_list)

    click.echo('Writing combined corpus to {0}'.format(out_filepath))
    with open(out_filepath, 'w') as out_file:
        for song in file_list:
            lines = open(song,'r').readlines()
            out = lines # do no preprocessing
            out_file.writelines(out)

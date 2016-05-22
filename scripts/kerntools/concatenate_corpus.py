import click
import glob
import os

from constants import BACHBOT_DIR

@click.command()
@click.option('--file-list', help='A text file where each line is a path to an input kern file.')
@click.option('--remove-metadata', default=True, help='Strips kern metadata header information.')
@click.option('--out-dir', default=BACHBOT_DIR + '/scratch',
        help='The directory to write the processed and concatenated corpus (with same filename as the file-list).')
def concatenate_corpus(file_list, remove_metadata, out_dir):
    """Preprocesses raw kern files and concatenates into corpus."""
    if not file_list:
        file_list = glob.glob('{0}/corpus/Bach+Johann/*.krn'.format(BACHBOT_DIR))
        out_filepath = out_dir + '/Bach+Johann.txt'
    else:
        file_list = open(file_list, 'r').readlines()
        out_filepath = out_dir + '/' + os.path.basename(file_list)

    click.echo('Writing combined corpus to {0}'.format(out_filepath))
    if not os.path.exists(os.path.dirname(out_filepath)):
        os.makedirs(os.path.dirname(out_filepath))

    with open(out_filepath, 'w+') as out_file:
        for song in file_list:
            lines = open(song,'r').readlines()
            out = preprocess_kern(lines, remove_metadata)
            out_file.writelines(out)

def preprocess_kern(lines, remove_metadata):
    """Preprocesses kern file format, replacing measure numbers with '@' and optionally removing header.

    :lines: List[String] : the lines of a kern file
    :remove_metadata: Boolean : removes the header (e.g. tempo, key signature, time signature)
    :returns: List[String] : the processed lines
    """
    REP='@\n' # delimiters between measures
    out = []
    found_first = False
    for l in lines:
        if l.startswith('='):
            ## new measure, replace the measure with the @ sign, not part of humdrum
            out.append(REP)
            found_first = True
            continue
        if not found_first:
            ## keep going until we find the end of the header
            continue
        if l.startswith('!'):
            ## ignore comments
            continue
        out.append(l)
    return out

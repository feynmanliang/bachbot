import click
import glob
import os.path

from music21 import *

from constants import BACHBOT_DIR

@click.command()
@click.option('--file-list', type=click.File('rb'),
        help='A text file where each line is a path to an input kern file.')
@click.option('--out-dir', default=BACHBOT_DIR + '/scratch', type=click.Path(exists=True),
        help='The directory to write the *-mono.kern files to.')
def extract_melody(file_list, out_dir):
    """Extracts a monophonic melody voices from a list of kern file.

    We define the melody line to be the [0]th spine in the parsed music21 score.
    """

    if not file_list:
        file_list = glob.glob('{0}/corpus/Bach+Johann/chor*.krn'.format(BACHBOT_DIR))
    else:
        file_list = file_list.readlines()

    for f in file_list:
        score = converter.parse(f)
        melodyPart = score.parts.stream()[0]

        fname = os.path.splitext(os.path.basename(f))[0]
        outPath = out_dir + '/{0}-mono.xml'.format(fname)
        print("Writing to {0}".format(outPath))
        try:
            score.write(fmt='musicxml', fp=outPath)
        except StreamException:
            print "Unexpected error:", sys.exc_info()[0]
            print "Skipping " + outPath
            continue


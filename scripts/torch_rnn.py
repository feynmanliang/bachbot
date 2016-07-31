import click
import codecs
import json
import subprocess

from music21 import note, stream

from constants import *

@click.command()
@click.option('-i', '--infile', type=click.File('rb'), default=SCRATCH_DIR + '/concat_corpus.txt')
@click.option('-o', '--outdir', type=click.Path(exists=True), default=SCRATCH_DIR)
def make_h5(infile, outdir):
    """Encodes corpus for torch-rnn."""
    fileName=os.path.splitext(os.path.basename(infile.name))[0]

    # preprocess data (tokenize store in hdf5)
    infile_path = os.path.abspath(infile.name)
    print 'Processing corpus at: ' + infile_path
    print 'Outputting to: ' + outdir + '/' + fileName + '{h5,json}'
    subprocess.call(' '.join([
        'python',
        '~/torch-rnn/scripts/preprocess.py',
        '--input_txt', infile_path,
        '--output_h5', outdir + '/' + fileName + '.h5',
        '--output_json', outdir + '/' + fileName + '.json'
    ]), shell=True)

@click.command()
@click.option('-i', '--input_h5', type=click.Path(exists=True), default=SCRATCH_DIR + '/concat_corpus.h5')
@click.option('-c', '--checkpoint_dir', type=click.Path(), default=SCRATCH_DIR + '/checkpoints')
def train(input_h5, checkpoint_dir):
    """Trains torch-rnn model. Alias to bachbot/scripts/torchrnn/train.zsh."""
    subprocess.call(' '.join([
        BACHBOT_DIR + '/scripts/torchrnn/train.zsh',
        input_h5,
        checkpoint_dir
    ]), shell=True)

@click.command()
@click.argument('checkpoint', type=click.Path(exists=True), required=True)
@click.option('-t', '--temperature', type=float, default=0.9)
@click.option('-s', '--start-text-file', type=click.Path(exists=True), help='Primer UTF file')
def sample(checkpoint, temperature, start_text_file):
    """Samples torch-rnn model. Calls bachbot/scripts/torchrnn/sample.zsh."""
    if not start_text_file:
        start_text = START_DELIM
    else:
        start_text = codecs.open(start_text_file, 'r', 'utf8').read()[:320]
    subprocess.call(
            BACHBOT_DIR + '/scripts/torchrnn/sample.zsh {0} {1} {2}'.format(checkpoint, temperature, start_text),
            shell=True)

import click

import codecs
import csv
import numpy as np
import os.path
import glob
from music21 import corpus
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

from datasets import build_vocabulary
from constants import *


@click.group()
def score():
    """Tools for scoring MusicXML outputs."""
    pass

@click.command()
@click.option('-o', '--out-file', type=click.File('wb'), default=open(SCRATCH_DIR + '/harm_results.csv', 'wb'))
@click.pass_context
def harm_error_rate(ctx, out_file):
    num_correct = defaultdict(Counter)
    num_total = defaultdict(Counter)

    error_types = ['TER', 'FER']

    for output in glob.glob(os.path.join(SCRATCH_DIR, 'harm_out', '*-mask-*.utf'))[:30]:
        harm_fname = os.path.basename(output)
        orig_fname, parts_mask = extract_BWV(harm_fname)
        masked = os.path.join(SCRATCH_DIR, harm_fname)
        reference = os.path.join(SCRATCH_DIR, orig_fname)
        error_rates = ctx.invoke(
                harm_error_rate_single,
                output=output, masked=masked, reference=reference)

        for error_type, (n_correct, n_total) in zip(error_types, error_rates):
            num_correct[error_type][parts_mask] += n_correct
            num_total[error_type][parts_mask] += n_total

    outwriter = csv.writer(out_file)
    for t in error_types:
        for parts_mask in num_correct[t]:
            outwriter.writerow((t, parts_mask, float(num_correct[t][parts_mask]) / num_total[t][parts_mask]))
    print 'Wrote results to {}'.format(out_file.name)

def extract_BWV(fname):
    parts = fname.split('-')
    mask_idx = parts.index('mask')
    orig_fname = '-'.join(parts[:mask_idx]) + '-nomask-fermatas.utf'
    parts_mask = ''.join(map(lambda x: x[0], parts[mask_idx+1:-1]))
    return orig_fname, parts_mask

@click.command()
@click.argument('output', required=True, type=click.Path(exists=True))
@click.argument('masked', required=True, type=click.Path(exists=True))
@click.argument('reference', required=True, type=click.Path(exists=True))
def harm_error_rate_single(output, masked, reference):
    txt_to_utf, _= build_vocabulary()
    n_tok_correct = 0
    n_tok_total = 0
    n_frame_correct = 0
    n_frame_total = 0

    out_fd = codecs.open(output, 'r', 'utf8').read()
    masked_fd = codecs.open(masked, 'r', 'utf8').read()
    ref_fd = codecs.open(reference, 'r', 'utf8').read()
    frame_buffer_out = u''
    frame_buffer_ref = u''
    for o, (m, r) in zip(out_fd, zip(masked_fd, ref_fd)):
        if r in [txt_to_utf[CHORD_BOUNDARY_DELIM], END_DELIM] and len(frame_buffer_out) > 0: # new frame
            n_frame_total += 1
            if frame_buffer_ref == frame_buffer_out:
                n_frame_correct += 1
            frame_buffer_out = u''
            frame_buffer_ref = u''
        elif m == BLANK_MASK_UTF: # the model is making a prediction
            n_tok_total += 1
            frame_buffer_out += o
            frame_buffer_ref += r
            if o == r:
                n_tok_correct += 1
    return (n_tok_correct, n_tok_total), (n_frame_correct, n_frame_total)

@click.command()
@click.option('-i', '--input-path', required=True, type=click.File('rb'))
def pca_metric(input_path):
    """Computes distance to Bach centroid in 2D PCA."""
    out_fp = SCRATCH_DIR + '/score-compute_PCs.npy'
    if not os.path.exists(out_fp):
        _compute_PCs(out_fp)
    PCs = np.fromfiile(out_fp)
    print PCs
    # TODO: find centroid of Bach cluster
    # TODO: compute and return neuclidian distance of PC projectio nof input_path file to Bach centroid


def _compute_PCs(out_fp):
    """Computes pitch class per measure principal components."""
    bachBundle = corpus.search('bwv')
    bachBundle = bachBundle.search('4/4')

    # NOTE: we should refactor this into a separate helper function: music21 Stream -> Pitch class histogram
    index =0
    data = {}
    for n in range(len(bachBundle)):
        data[n] = {}
        for i in range(30,100):
            data[n][i] = 0

    for n in range(len(bachBundle)):
        myPiece = bachBundle[n].parse()

        for m in myPiece.flat.getElementsByClass('Note'):
            data[n][m.midi] +=1

        print 'Number %i' % n

    new_data = np.array([data[0].values()]).astype(np.float64)
    new_data /= np.sum(new_data)

    for index in range(len(bachBundle)):
        temp = np.array([data[index].values()]).astype(np.float64)
        temp /= np.sum(temp)
        new_data =  np.concatenate((new_data,  temp)  , axis=0)

    print 'Statistics gathered!'
    save = new_data

###############################################################################
    bachBundle = corpus
    bachBundle = bachBundle.search('4/4')

    index =0
    data = {}
    for n in range(700, 2500):
        data[n] = {}
        for i in range(30,100):
            data[n][i] = 0

    for n in range(700, 2500):
        myPiece = bachBundle[n].parse()

        for m in myPiece.flat.getElementsByClass('Note'):
                data[n][m.midi] +=1

        print 'Number %i' % n

    new_data = np.array([data[700].values()])
    new_data /= np.sum(new_data)

    for index in range(700, 2500):
        temp = np.array([data[index].values()]).astype(np.float64)
        temp /= np.sum(temp)

        new_data =  np.concatenate( (new_data,  temp )  , axis=0)

    print 'Statistics gathered!'

    X = new_data
    d = np.concatenate((save,X))
    n_components=2
    pca = PCA(n_components=n_components).fit(d)
    pca.components_.tofile(out_fp)

    print '{} PCs written to {}'.format(n_components, out_fp)

map(score.add_command, [
    pca_metric,
    harm_error_rate
])

import click

import numpy as np
import os.path

from music21 import corpus
from sklearn.decomposition import PCA

from constants import *


@click.group()
def score():
    """Interface for scoring MusicXML outputs."""
    pass

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
    pca_metric
])

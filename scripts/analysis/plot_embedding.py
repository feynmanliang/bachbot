#!/usr/bin/python

import h5py
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json
import codecs

from music21 import *

data = dict()
for embed in ['input', 'char_embeddings', 'embedding-lstm1', 'embedding-lstm2']:
    fname = embed + '.h5'
    f = h5py.File('./' + fname, 'r')
    data[embed] = f['home/fl350/data/' + fname].value

utf_to_txt = json.load(codecs.open('./utf_to_txt.json', 'rb', 'utf-8'))
corpus_vocab = json.load(codecs.open('./concat_corpus.json', 'rb', 'utf-8'))

data['input'] = data['input'].squeeze()
x = []
for idx in list(data['input']):
    txt = utf_to_txt[corpus_vocab['idx_to_token'][str(idx)]]
    if len(txt) > 5: # NOTE: hacky way to tell if we have a note, but works...
        midi, tied = eval(txt)
        n = note.Note()
        n.pitch.midi = midi
        #data['input'].append((n.pitch.nameWithOctave, tied))
        x.append(n.pitch.nameWithOctave)
    else:
        x.append(txt)
data['input'] = np.array(x)

X = data['char_embeddings']


# PCA plot
pca = PCA(n_components=2)
pca_embed = pca.fit_transform(X)

fig = plt.figure(figsize=(15, 8))
plt.title("PCA on character embeddings", fontsize=14)
plt.scatter(pca_embed[:,0], pca_embed[:,1], cmap=plt.cm.rainbow)

for label, x, y in zip(data['input'], pca_embed[:, 0], pca_embed[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (0, 0),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round', fc = 'grey', alpha = 0.25),)
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid()
plt.savefig('PCA-notes.png')
plt.show()

# tSNE plot
pca = PCA(n_components=50)
pca_data = pca.fit_transform(X)
model = TSNE(n_components=2, random_state=0)
tsne_embed = model.fit_transform(X)
tsne_embed_pca = model.fit_transform(pca_data)

fig = plt.figure(figsize=(15, 8))
plt.title("tSNE on character embeddings", fontsize=14)
plt.scatter(tsne_embed[:,0], tsne_embed[:,1], cmap=plt.cm.rainbow)

for label, x, y in zip(data['input'], tsne_embed[:, 0], tsne_embed[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (0, 0),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round', fc = 'grey', alpha = 0.25),)
        #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.xlabel('tSNE dim 1')
plt.ylabel('tSNE dim 2')
plt.grid()
plt.savefig('tSNE-notes.png')
plt.show()

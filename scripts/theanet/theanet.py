#!/usr/bin/env python

import climate
import codecs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import theanets

import utils

from constants import *

climate.enable_default_logging()

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

VAL_FRACTION = 0.9

# path = '/home/fl350/torch-rnn/data/tiny-shakespeare.txt'
path = SCRATCH_DIR + '/concat_corpus.txt'

with codecs.open(path, 'r', 'utf-8') as handle:
    file_data = handle.read().lower()[:3000]
    text = theanets.recurrent.Text(file_data[:int(VAL_FRACTION*len(file_data))])
    text_val = theanets.recurrent.Text(file_data[int(VAL_FRACTION*len(file_data)):])

text_val.alpha = text.alpha

seed = text.encode(text.text[0])
for i, layer in enumerate((
        dict(form='rnn', activation='relu'),
        dict(form='gru', activation='sigmoid'),
        dict(form='scrn', activation='linear'),
        dict(form='lstm'),
        dict(form='mrnn', activation='sigmoid', factors=len(text.alpha)),
        dict(form='clockwork', activation='relu', periods=(1, 2, 4, 8, 16)),
    )):
    losses_t = []
    losses_v = []
    layer.update(size=130)
    net = theanets.recurrent.Classifier([
        1 + len(text.alpha), 64, layer, 1 + len(text.alpha)])
    for tm_t, tm_v in net.itertrain(
            text.classifier_batches(50, 64), # dimensions: minibatch, time
            text_val.classifier_batches(50, 64),
            algo='rmsprop',
            min_improvement=0.99,
            max_gradient_norm=5,
            validate_every=1,
            patience=5,
            learning_rate=0.01):
        if np.isnan(tm_t['loss']):
            break
        print(u'{}|{} ({:.1f}%)'.format(
            text.decode(seed),
            text.decode(net.predict_sequence(seed, 10)),
            100 * tm_v['acc']))
        losses_t.append(tm_t['loss'])
        losses_v.append(tm_v['loss'])
        print(tm_t['acc'])
        print(tm_v['acc'])
        print(tm_t['loss'])
        print(tm_v['loss'])

    plt.subplot(2,1,1)
    plt.plot(losses_t, label=layer['form'], alpha=0.7, color=COLORS[i])
    plt.subplot(2,1,2)
    plt.plot(losses_v, label=layer['form'], alpha=0.7, color=COLORS[i])

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_position(('outward', 6))
plt.gca().spines['left'].set_position(('outward', 6))

plt.gca().set_ylabel('Loss')
plt.gca().set_xlabel('Training Epoch')
plt.gca().grid(True)

plt.legend()
plt.show()
plt.savefig('plot.png')

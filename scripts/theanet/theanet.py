#!/usr/bin/env python

import climate
import collections
import codecs
import matplotlib.pyplot as plt
import numpy as np
import pickle
import theanets

import utils

from constants import *

climate.enable_default_logging()

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

TRAIN_FRACTION = 0.9

#path = '/home/fl350/torch-rnn/data/tiny-shakespeare.txt'
path = SCRATCH_DIR + '/concat_corpus.txt'

with open(path, 'r') as handle:
    file_data = unicode(handle.read(), 'utf8')
    text = theanets.recurrent.Text(file_data[:int(TRAIN_FRACTION*len(file_data))])
    text_val = theanets.recurrent.Text(file_data[int(TRAIN_FRACTION*len(file_data)):])

text_val.alpha = text.alpha

losses_t = collections.defaultdict(list)
losses_v = collections.defaultdict(list)

seed = text.encode(text.text[0])
for i, layer in enumerate((
        dict(form='rnn', activation='sigmoid'),
        dict(form='gru', activation='sigmoid'),
        dict(form='scrn', activation='linear'),
        dict(form='lstm'),
        dict(form='mrnn', activation='sigmoid', factors=len(text.alpha)),
        dict(form='clockwork', activation='sigmoid', periods=(1, 2, 4, 8, 16)),
    )):
    form = layer['form']
    layer.update(size=130)
    net = theanets.recurrent.Classifier([
        1 + len(text.alpha), 64, layer, 1 + len(text.alpha)])
    train_iter = 0
    for tm_t, tm_v in net.itertrain(
            text.classifier_batches(50, 64), # dimensions: minibatch, time
            text_val.classifier_batches(50, 64),
            algo='rmsprop',
            min_improvement=0.99,
            max_gradient_norm=5,
            validate_every=1,
            patience=30,
            learning_rate=0.01):
        train_iter += 1
        if np.isnan(tm_t['loss']):
            break
        print(u'{}|{} ({:.1f}%)'.format(
            text.decode(seed),
            text.decode(net.predict_sequence(seed, 10)),
            100 * tm_v['acc']))
        losses_t[form].append(tm_t['loss'])
        losses_v[form].append(tm_v['loss'])
        print(tm_t['acc'])
        print(tm_v['acc'])
        print(tm_t['loss'])
        print(tm_v['loss'])
        net.save(SCRATCH_DIR + '/theanets_cv/{}_{}.pkl'.format(form, train_iter))

pickle.dump(losses_t, open(SCRATCH_DIR + '/theanets_cv/losses_t.pkl', 'wb'))
pickle.dump(losses_v, open(SCRATCH_DIR + '/theanets_cv/losses_v.pkl', 'wb'))

for i, form in enumerate(losses_t.keys()):
    if form in ['scrn']:
        continue
    plt.subplot(2,1,1)
    plt.plot(losses_t[form], label=form, alpha=0.7, color=COLORS[i])

    plt.subplot(2,1,2)
    plt.plot(losses_v[form], label=form, alpha=0.7, color=COLORS[i])

for subplot in range(1,3):
    plt.subplot(2,1,subplot)

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

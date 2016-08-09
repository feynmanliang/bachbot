#!/usr/bin/env python
import glob
import codecs


files = glob.glob('/home/fl350/bachbot/scratch/BWV-*-nomask-fermatas.txt')
train_lines = []
val_lines = []
for i,fp in enumerate(files):
    with codecs.open(fp, 'r', 'utf-8') as fd:
        lines = map(
                lambda x: filter(lambda c: c not in ' \n', x),
                fd.readlines())
        if i <= int(0.9 * len(files)):
            train_lines.append(' '.join(lines))
        else:
            val_lines.append(' '.join(lines))

with open('corpus_train.txt', 'w') as fd:
    fd.write('\n'.join(train_lines))

with open('corpus_val.txt', 'w') as fd:
    fd.write('\n'.join(val_lines))

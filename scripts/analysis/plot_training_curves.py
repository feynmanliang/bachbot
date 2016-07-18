#!/usr/bin/python

import h5py
import numpy as np

import re

import matplotlib.pyplot as plt

train = []
val = []
epoch = 0
i = 0
for line in open('train.log', 'r'):
    if re.match(r"^Epoch", line):
        sp = line.split()
        epoch = float(sp[1])
        i = float(sp[6])
        loss = float(sp[-1])
        train.append((epoch, i, loss))
    else:
        val.append((epoch, i, float(line.split()[-1])))

train = np.array(train)
val = np.array(val)

plt.figure()
plt.title('Training curves')
plt.plot(
        train[:,0], 2 ** train[:,2], # exponentiate cross-entropy loss to get perplexity
        val[:,0], 2 ** val[:,2])
plt.legend(['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.grid()
plt.savefig('training-curves.png')
plt.show()

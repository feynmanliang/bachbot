import click
from music21 import *
import codecs
import numpy as np

from progress.bar import Bar

from keras.callbacks import EarlyStopping, TensorBoard, ProgbarLogger, ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge

import sys
import random

from constants import *

@click.group()
def interpolation():
    """Interface for interpolation experiments."""
    pass

np.random.seed(42)

max_length = 100
embedding_dim = 128
hidden_size = 64
batch_size = 32
nb_epoch = 50
N_samples = 10000

@click.command()
def train():
    X, y, vocab_size, _, _ = prepare_data()
    model = build_model(vocab_size)

    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_split=0.1,
            callbacks = [ ProgbarLogger(),
                ModelCheckpoint(SCRATCH_DIR + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                TensorBoard(log_dir='./logs', histogram_freq=0.1) ])

def prepare_data():
    text = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())
    chars = sorted(list(set(text)) + [PADDING])
    vocab_size = len(chars)
    print('vocab size: {}'.format(vocab_size))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # extract sliding window contexts, respecting file boundaries `END_DELIM`
    contexts = []
    next_chars = []
    curr_context = []
    for char in text:
        if len(curr_context) == max_length: # add a valid context + next_char
            contexts.append(curr_context)
            next_chars.append(char)
        if char == END_DELIM: # clear context if END_DELIM
            curr_context = []
        elif len(curr_context) < max_length: # add padded version, keep adding to context if < max_length
            contexts.append([PADDING] * (max_length - len(curr_context)) + curr_context)
            next_chars.append(char)
            curr_context.append(char)
        else: # slide context fowards
            curr_context = curr_context[1:] + [char]

    # TODO: should we pad with zeros to allow for initializing context < max_length?
    # NOTE: torch-rnn doesn't apepar to do so...

    # vectorize
    X = np.zeros((len(contexts), max_length), dtype=np.int32)
    y = np.zeros((len(contexts), len(chars)), dtype=np.bool)
    for i, context in enumerate(contexts):
        for t, char in enumerate(context):
            X[i, t] = char_indices[char]
        y[i, char_indices[next_chars[i]]] = 1
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    return X, y, vocab_size, char_indices, indices_char

def build_model(vocab_size):
    print('Build model...')
    sequence = Input(shape=(max_length,), dtype='int32')
    embedded = Embedding(vocab_size, embedding_dim, input_length=max_length)(sequence)

    forwards = LSTM(hidden_size)(embedded)
    backwards = LSTM(hidden_size, go_backwards=True)(embedded)

    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    after_dp = Dropout(0.2)(merged)
    output = Dense(vocab_size, activation='softmax')(after_dp)
    model = Model(input=sequence, output=output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print "model.summary()"
    model.summary()
    return model

@click.command()
def sample():
    _, _, vocab_size, char_indices, indices_char = prepare_data()
    model = build_model(vocab_size)
    model.load_weights('scratch/weights.41-1.54.hdf5')

    def _sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    # generate samples
    # TODO: use START_DELIM as seed
    for temperature in [1.0]:
    #for temperature in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- temperature:', temperature)

        generated = ''
        sentence = PADDING*(max_length-1) + START_DELIM
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        # TODO: outfile option
        with open('out', 'wb') as fd:
            for i in Bar('Sampling').iter(range(N_samples)):
                # vectorize sentence context
                x = np.zeros((1, max_length))
                for t, char in enumerate(sentence):
                    x[0, t] = char_indices[char]

                preds = model.predict(x, verbose=0)[0]
                next_index = _sample(preds, temperature)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char
            fd.write(generated)

map(interpolation.add_command, [
    train,
    sample
])


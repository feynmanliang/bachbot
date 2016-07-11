import click
from music21 import *
import codecs
import numpy as np

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

@click.command()
def train():
    np.random.seed(42)

    max_length = 100

    embedding_dim = 128
    hidden_size = 64

    batch_size = 32
    epochs = 50

    # TODO: use everything
    text = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())[:1000]
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
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
        elif len(curr_context) < max_length: # keep adding to context if < max_length
            curr_context.append(char)
        else: # slide context fowards
            curr_context = curr_context[1:] + [char]


    # zero pad X, vectorize y
    X = np.zeros((len(contexts), max_length), dtype=np.int32)
    y = np.zeros((len(contexts), len(chars)), dtype=np.bool)
    for i, context in enumerate(contexts):
        for t, char in enumerate(context):
            X[i, t] = char_indices[char]
        y[i, char_indices[next_chars[i]]] = 1
    print('X shape:', X.shape)
    print('y shape:', y.shape)

    print('Build model...')
    sequence = Input(shape=(max_length,), dtype='int32')
    embedded = Embedding(len(chars), embedding_dim, input_length=max_length)(sequence)

    forwards = LSTM(hidden_size)(embedded)
    backwards = LSTM(hidden_size, go_backwards=True)(embedded)

    merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
    after_dp = Dropout(0.2)(merged)
    output = Dense(len(chars), activation='softmax')(after_dp)

    model = Model(input=sequence, output=output)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    # train the model, output generated text after each iteration
    for iteration in range(1, epochs):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=batch_size, nb_epoch=1)

        start_index = random.randint(0, len(text) - maxlen - 1)

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- temperature:', temperature)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


map(interpolation.add_command, [
    train
])


import click
import glob
import json
import numpy as np

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, Input, Embedding, LSTM, Dense, merge, Lambda, BatchNormalization
from keras.models import Sequential, Model, model_from_json
from keras.preprocessing.text import Tokenizer

from music21 import *

from constants import *
from torch_rnn import read_utf8
from concatenate_corpus import concatenate_corpus

@click.group()
def keras():
    """Interface for working with keras/tensorflow models."""
    pass

@click.pass_context
def prepare(ctx, maxlen):
    """Prepares Soprano 4/4 Major key pitch classes corpus."""
    if len(glob.glob(SCRATCH_DIR + '/*soprano-mono.utf')) == 0:
        ctx.invoke(prepare_mono_all, use_pitch_classes=True)
    if not os.path.exists(SCRATCH_DIR + '/concat_corpus.txt'):
        ctx.invoke(concatenate_corpus,
                files=glob.glob(SCRATCH_DIR + '/*soprano-mono.utf'),
                output=open(SCRATCH_DIR + '/concat_corpus.txt', 'wb'))

    texts = read_utf8(
            SCRATCH_DIR + '/concat_corpus.txt',
            json.loads(open(SCRATCH_DIR + '/utf_to_txt.json', 'rb').read()))
    V = len(reduce(lambda x, y: set(x).union(set(y)), texts))

    tok = Tokenizer(nb_words=V, filters='', char_level=True)
    tok.fit_on_texts(texts)
    data = tok.texts_to_sequences(texts)

    sentences, next_chars = _sliding_window(data, maxlen=maxlen, step=2)
    X, y = _vectorize_window(sentences, next_chars, maxlen=maxlen, V=V)

    return tok, X, y

def _sliding_window(data, maxlen, step):
    sentences = []
    next_chars = []
    for score in data:
        for i in range(0, len(score) - maxlen, step):
            sentences.append(score[i: i + maxlen])
            next_chars.append(score[i + maxlen])
    return sentences, next_chars

def _vectorize_window(sentences, next_chars, maxlen, V):
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, V), dtype=np.bool)
    y = np.zeros((len(sentences), V), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char] = 1
        y[i, next_chars[i]] = 1
    return X, y

@click.command()
@click.option('--maxlen', type=int, default=40, help='Length of context used for inputs')
@click.pass_context
def train_lstm(ctx, maxlen):
    """build the model: 2 stacked LSTM."""
    tok, X, y = prepare(maxlen)
    V = tok.nb_words
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, V)))
    model.add(BatchNormalization())
    model.add(LSTM(128, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(V))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='loss', patience=5) # NOTE: we intentionally overfit training set here
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)

    model.fit(X, y,
	    nb_epoch=30, batch_size=32,
	    validation_split=0.1,
	    callbacks=[early_stopping, tensorboard])

    open('model-lstm.json', 'wb').write(model.to_json())
    model.save_weights('model-lstm_weights.h5', overwrite=True)

    return model

@click.command()
@click.argument('model_json', type=click.File('rb'))
@click.argument('model_h5', type=click.FilePath(exists=True))
@click.option('temperature', type=float, default=1.0, help='Sampling temperature')
def sample_lstm(model_json, model_h5):
    """Samples a trained LSTM and outputs to stdout."""
    model = model_from_json(model_json.read())
    model.load_weights(model_h5)
    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()

    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        start_sentence = sentences[np.random.randint(0,len(sentences))]
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = list()
            sentence = start_sentence
            generated += sentence
            print('----- Generating with seed: "' + str(sentence) + '"')

            for i in range(400):
                x = np.zeros((1, maxlen, tok.nb_words))
                print sentence
                for t, char in enumerate(sentence):
                    x[0, t, char] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                generated += next_index

                sentence = sentence[1:] + next_index

            print dir(tok)
            sys.stdout.write(tok.index_to_sequence(generated))
            sys.stdout.flush()
            print()
#def make_skipgrams(data, V):
#    X, Y = list(), list()
#    for d in data:
#        x, y = skipgrams(d, V, window_size=3)
#        x = map(np.array, x)
#        y = map(np.array, y)
#        X.extend(x)
#        Y.extend(y)
#    return np.array(X), np.array(Y)
#
#tok, data = prepare()
#V = len(tok.word_counts) + 1
#X, Y = make_skipgrams(data, V)
#X_train, Y_train = X, Y # NOTE: no test split
#
#wordvec_size = 64
#
#def train_skipgram():
#        raw_in = Input(shape=(2,), name='raw_in', dtype='int32')
#        raw_x = Lambda(lambda x: x[:,0], input_shape=(2,), output_shape=(1,))(raw_in)
#        raw_other = Lambda(lambda x: x[:,1], input_shape=(2,), output_shape=(1,))(raw_in)
#
#        embedding = Embedding(output_dim=wordvec_size, input_dim=V)
#        x = embedding(raw_x)
#        other = embedding(raw_other)
#        diff = merge([x, other], mode=lambda t: t[0] - t[1], output_shape=(wordvec_size,))
#        sg_loss = Dense(1, activation='sigmoid', name='skip_gram')(diff)
#
#        # Compile and fit
#        model = Model(input=[raw_in], output=[sg_loss])
#        model.compile(optimizer='adagrad',
#                      loss={'skip_gram': 'binary_crossentropy'},
#                      loss_weights={'skip_gram': 1.0},
#                      metrics=['accuracy'])
#
#        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#        model.fit({'raw_in': X_train},
#                  {'skip_gram': Y_train},
#                  nb_epoch=30, batch_size=32,
#                  validation_split=0.1,
#                  callbacks=[early_stopping])
#
#        open('model-sg.json', 'wb').write(model.to_json())
#        model.save_weights('model-sg_weights.h5', overwrite=True)
##train_skipgram()
#
##model = load('model-sg')
## model.compile(optimizer='rmsprop',
##         loss={'skip_gram': 'categorical_crossentropy'},
##         loss_weights={'skip_gram': 1.0},
##         metrics=['accuracy'])
#

map(keras.add_command, [
    train_lstm,
    sample_lstm
])


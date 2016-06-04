import click
import copy
import cPickle
import glob
import json
import numpy as np

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, Input, Embedding, LSTM, Dense, merge, Lambda, BatchNormalization
from keras.models import Sequential, Model, model_from_json
from keras.preprocessing.text import Tokenizer

from music21 import *

from constants import *
from corpus_utils import concatenate_corpus, read_utf8, write_monophonic_part

@click.group()
def keras():
    """Interface for working with keras/tensorflow models."""
    pass

@click.pass_context
def prepare(ctx, maxlen):
    """Prepares Soprano 4/4 Major key pitch classes corpus."""
    if len(glob.glob(SCRATCH_DIR + '/*soprano-mono.utf')) == 0:
        ctx.invoke(prepare_mono_all, use_pitch_classes=False)
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
@click.option('--output-json', default=SCRATCH_DIR + '/model-lstm.json', type=click.File('wb'))
@click.option('--output-h5', default=SCRATCH_DIR + '/model-lstm_weights.h5', type=click.Path())
@click.option('--output-tok', default=SCRATCH_DIR + '/model-lstm_tok.pickle', type=click.File('wb'))
@click.pass_context
def train_lstm(ctx, maxlen, output_json, output_h5, output_tok):
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

    output_json.write(model.to_json())
    model.save_weights(output_h5, overwrite=True)
    cPickle.dump(tok, output_tok)

    return model

@click.command()
@click.option('--model_json', default=SCRATCH_DIR + '/model-lstm.json', type=click.File('rb'))
@click.option('--model_h5', default=SCRATCH_DIR + '/model-lstm_weights.h5', type=click.Path(exists=True))
@click.option('--model_tok', default=SCRATCH_DIR + '/model-lstm_tok.pickle', type=click.File('rb'))
@click.option('--start_txt', default=SCRATCH_DIR + '/20.7-major-soprano-mono.txt', type=click.File('r'),
        help='Uses the first `maxlen` notes of the provided mono text to prime the RNN')
@click.option('--out_prefix', default=OUT_DIR + '/sample', type=str)
def sample_lstm(model_json, model_h5, model_tok, start_txt, out_prefix):
    """Samples a trained LSTM and outputs to stdout."""
    tok = cPickle.load(model_tok)
    V = tok.nb_words
    model = model_from_json(model_json.read())
    model.load_weights(model_h5)
    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()
    maxlen = model.layers[0].input_shape[1]

    # prime RNN with existing chorale
    start_sentence = start_txt.read().split('\n')[:maxlen]

    # helper function to sample an index from a probability array
    def sample(a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    index_words = {v:k for k,v in tok.word_index.items()}
    for iteration in range(1, 5):
        print()
        print('-' * 50)
        print('Iteration', iteration)

        for temperature in [0.8, 1.3, 1.8]:
            print()
            print('----- temperature:', temperature)

            generated = copy.copy(start_sentence)
            sentence = map(tok.word_index.get, start_sentence[-maxlen:])
            print('----- Generating with seed: "' + str(start_sentence) + '"')

            for i in range(100):
                x = np.zeros((1,maxlen,V))
                x[0,:,:] = tok.texts_to_matrix([sentence])

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_word = index_words[next_index]
                generated.append(next_word)

                print(next_word)
                sentence = sentence[1:] + next_index
            print(generated)

            out_fp = '{0}-{1}-{2}.xml'.format(out_prefix, temperature, iteration)
            write_monophonic_part(generated, out_fp)

@click.command()
@click.pass_context
def train_discrim(ctx):
    from keras.layers import Input, Embedding, LSTM, Dense, merge
    from keras.models import Model

    tok, X, y = prepare_discrim()

    score_in = Input(shape=(X.shape[1],), dtype='int32', name='score_in')
    x = Embedding(output_dim=64, input_dim=tok.nb_words, input_length=X.shape[1])(score_in)
    lstm_out = LSTM(32)(x)
    x = Dense(64, activation='relu')(lstm_out)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(input=[score_in], output=[output])
    model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
    model.fit(X, y,
            nb_epoch=30, batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping, tensorboard])

    open('model.json','wb').write(model.to_json())
    model.save_weights('model.h5', overwrite=True)
    cPickle.dump(tok, 'tok.pickle')


@click.pass_context
def prepare_discrim(ctx):
    if len(glob.glob(SCRATCH_DIR + '/*soprano-mono.utf')) == 0:
        ctx.invoke(prepare_mono_all, use_pitch_classes=False)
    data = dict()
    for mode in ['major','minor']:
        fp = SCRATCH_DIR + '/concat_corpus-{0}.txt'.format(mode)
        if not os.path.exists(fp):
            ctx.invoke(concatenate_corpus,
                    files=glob.glob(SCRATCH_DIR + '/*{0}-soprano-mono.utf'.format(mode)),
                    output=open(fp, 'wb'))

        data[mode] = read_utf8(
                fp,
                json.loads(open(SCRATCH_DIR + '/utf_to_txt.json', 'rb').read()))

    # lazy view of all scores in single collection
    all_data = lambda: [score for mode in data for score in data[mode]]
    N = len(all_data())

    # tokenize
    V = len(reduce(lambda x,y: x|y, map(set, all_data())))
    tok = Tokenizer(nb_words=V, filters='', char_level=True)
    tok.fit_on_texts(all_data())

    seq_length = max(map(len, all_data())) # NOTE: implicit zero pad all sequences to fixed length
    X = np.zeros((N, seq_length), dtype=np.uint16)
    y = np.zeros((N,), dtype=np.bool)
    i = 0
    for mode in data:
        for tokenized_score in tok.texts_to_sequences(data[mode]):
            X[i,:len(tokenized_score)] = tokenized_score
            y[i] = mode == 'major'
            i += 1
    # shuffle training data
    idxs = np.random.permutation(len(X))
    return tok, X[idxs], y[idxs]

map(keras.add_command, [
    train_lstm,
    sample_lstm,
    train_discrim
])


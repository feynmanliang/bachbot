import click
import copy
import cPickle
import glob
import json
import numpy as np

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from music21 import *

from chorales import prepare_standard
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

@click.command()
@click.pass_context
def biaxial(ctx):
    # Params for feature generation
    # NOTE: changing these requires rerunning dataset `Xy` generation
    part_context_size = 16
    all_voices_context_size = 1

    # Params for model
    note_embedding_size = 16
    pc_embedding_size = 4
    beat_embedding_size = 4
    time_lstm_size = 16
    part_lstm_size = 8
    use_cache = True
    batch_size=2

    if not use_cache:
        # NOTE: BE CAREFUL ABOUT USING SUBSET, it will overwrite and require another regen
        dataset = ctx.invoke(prepare_standard, subset=False)
        vocab_size, Xy = _prepare_biaxial(dataset,
                use_cache=False,
                part_context_size=part_context_size,
                all_voices_context_size=all_voices_context_size)
        vocab_size += 1 # +1 for '0'
    else:
        vocab_size, Xy = _prepare_biaxial(None, use_cache=True)

    print 'vocab_size: {}\n Xy.shape: {}'.format(vocab_size, Xy.shape)

    X = Xy[:,:,:-1,]
    # TODO: y should account for articulations
    y = np.zeros(X.shape[0:3] + (vocab_size,))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,:,j,:] = to_categorical(Xy[i,:,j,0].astype(np.uint16), nb_classes=(vocab_size))
    # X indices are seore, part, time, feature => value
    # y indices are score, part, time, next_note => played?
    seqlen = X.shape[2]

    in_note = X[:,:,:,0].astype(np.uint16)
    in_art = X[:,:,:,1].astype(np.bool)
    in_pc = X[:,:,:,2].astype(np.uint16)
    in_freq = X[:,:,:,3].astype(np.float32) # NOTE: assumes no frequencies exactly 0
    in_beat = X[:,:,:,4].astype(np.uint16)
    in_part_context_pc = X[:,:,:,5:5+part_context_size].astype(np.uint16)
    in_all_voices_context_notes = X[:,:,:,5+part_context_size:-4*all_voices_context_size].astype(np.uint16)
    in_all_voices_context_art = X[:,:,:,-4*all_voices_context_size:].astype(np.bool)

    note_embedding = Embedding(
            output_dim=note_embedding_size,
            input_dim=2+vocab_size,
            input_length=X.shape[2],
            mask_zero=True)
    note_input = Input(
            shape=X.shape[1:3],
            dtype='int32',
            name='note')
    note_embed = TimeDistributed(note_embedding, name='note_embed')( note_input ) # distribute across parts (axis 1)

    pc_embedding = Embedding(
            output_dim=pc_embedding_size,
            input_dim=2+12,
            input_length=X.shape[2],
            mask_zero=True)
    pc_input = Input(
            shape=X.shape[1:3],
            dtype='int32',
            name='pitch_class')
    pc_embed = TimeDistributed(pc_embedding, name='pc_embed')( pc_input ) # distribute across parts (axis 1)

    art_input = Input(
            shape=X.shape[1:3],
            dtype='float32', # NOTE: upcast to float32
            name='articulated')
    freq_input = Input(
            shape=X.shape[1:3],
            dtype='float32',
            name='frequency')

    beat_embedding = Embedding(
            output_dim=beat_embedding_size,
            input_dim=2+8,
            input_length=X.shape[2],
            mask_zero=True)
    beat_input = Input(
            shape=X.shape[1:3],
            dtype='int32',
            name='beat')
    beat_embed = TimeDistributed(beat_embedding, name='beat_embed')( beat_input )

    part_context_input = Input(
            shape=X.shape[1:3] + (part_context_size,),
            dtype='int32',
            name='part_context')
    part_context_embed = TimeDistributed(TimeDistributed(Flatten()), name='part_context_embed')(
            Permute((1,3,2,4))( # permute axes: 0 = sample, 1 = part, 2 = time, 3 = context, 4 = embedded feature
                TimeDistributed(TimeDistributed(pc_embedding))( # distribute across parts and context_size
                    Permute((1,3,2))(part_context_input))))

    all_context_notes_input = Input(
            shape=X.shape[1:3] + (4*all_voices_context_size,),
            dtype='int32',
            name='all_context_notes')
    all_context_notes_embed = TimeDistributed(TimeDistributed(Flatten()), name='all_context_notes_embed')(
            Permute((1,3,2,4))( # permute axes: 0 = sample, 1 = part, 2 = time, 3 = context, 4 = embedded feature
                TimeDistributed(TimeDistributed(note_embedding))( # distribute across parts and context_size
                    Permute((1,3,2))(part_context_input))))
    all_context_art_input = Input(
            shape=X.shape[1:3] + (4*all_voices_context_size,),
            dtype='float32', # NOTE: upcast to float32
            name='all_context_articulated')

    time_lstm_input = merge([
        note_embed,
        pc_embed,
        Reshape(X.shape[1:3] + (1,))(art_input), # expand singleton axis so dimensions match
        Reshape(X.shape[1:3] + (1,))(freq_input),
        beat_embed,
        part_context_embed,
        all_context_notes_embed,
        all_context_art_input
        ], mode='concat', concat_axis=3,
        name='time_lstm_input')
    time_lstm0 = LSTM(time_lstm_size, return_sequences=True)
    time_lstm0_out = TimeDistributed(time_lstm0)( time_lstm_input ) # distribute across parts (axis 1)

    part_lstm0 = LSTM(part_lstm_size, return_sequences=True)
    part_lstm0_out = TimeDistributed(part_lstm0)( # now distribute across time (axis 1 after permute), run lstm along parts (axis 2)
            Permute((2,1,3))( # permute time to axis 1, parts to axis 2
                time_lstm0_out))

    softmax = Dense(vocab_size, activation='softmax')
    softmax_activations = TimeDistributed(TimeDistributed(softmax))( # distribute across time (1) and parts(2)
        part_lstm0_out)
    softmax_out0 = map(lambda i:
            Lambda(
                lambda x: x[:,:,i,:],
                output_shape=(seqlen, vocab_size),
                name='next_note{}'.format(i))( # extract part (now axis 2)
                    softmax_activations),
            range(X.shape[1]))

    model = Model(
            input=[
                note_input,
                pc_input,
                art_input,
                freq_input,
                beat_input,
                part_context_input,
                all_context_notes_input,
                all_context_art_input],
            output=softmax_out0)

    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    print "model.summary()"
    model.summary()

    model.fit(
        { 'note': in_note,
            'articulated': in_art,
            'pitch_class': in_pc,
            'frequency': in_freq,
            'beat': in_beat,
            'part_context': in_part_context_pc,
            'all_context_notes': in_all_voices_context_notes,
            'all_context_articulated': in_all_voices_context_art },
        {'next_note{}'.format(i):(y[:,i,:]) for i in range(4) },
        batch_size=batch_size, nb_epoch=25)


def _prepare_biaxial(dataset, use_cache=True, part_context_size=1, all_voices_context_size=2):
    """Prepares dataset following http://www.creativeai.net/posts/uvhEChAfmPKnG8swP.

    We include features for:
        * frequency
        * articulation (played at this time instant or held from previous time)
        * pitch class
        * beat (position within measure)
        * context of previous pitch classes for each part
        * context of previous notes/articulated? for all voices
    """
    fp = SCRATCH_DIR + '/biaxial_Xy.npy'
    if use_cache and os.path.exists(fp):
       Xy = np.load(fp)
       unique_notes = Xy[:,:,:,0] # NOTE: fragile indexing of note
       return len(set(unique_notes.flatten().tolist())), Xy
    else:
        # Each score is a 4 by T grid, X[i][j] is a vector with part_id, note, pitch, pitch_class, beat, etc
        unique_notes = set(['REST'])
        min_ql = 1
        longest_score_ql = 0
        for score in dataset:
            for part in score.parts:
                min_ql = min(min_ql, min(map(lambda note: note.duration.quarterLength, part.flat.notesAndRests)))
                longest_score_ql = max(longest_score_ql, part.duration.quarterLength)
                unique_notes = unique_notes | set(map(lambda x: x.nameWithOctave, part.flat.getElementsByClass('Note')))

        # choose time step so no loss in duration information
        #frames_per_crotchet = FRAMES_PER_CROTCHET
        frames_per_crotchet = int(1 / min_ql)
        max_num_frames = frames_per_crotchet * longest_score_ql

        # encode into feature matrix
        part_id_to_idx = {id:idx for idx,id in enumerate(['Soprano', 'Alto', 'Tenor', 'Bass'])}
        note_name_to_idx = {name:idx for idx,name in enumerate(unique_notes)}

        d = 5 + part_context_size + 4*2*all_voices_context_size # NOTE: change if # features increase
        X_all = np.zeros((len(dataset), 4, max_num_frames, d))
        # TODO: make indexing less fragile by using namedtuple, DataTable, pandas, etc
        for score_idx, score in enumerate(dataset):
            print 'Featurizing {} out of {}'.format(score_idx+1, len(dataset))
            num_frames = int(frames_per_crotchet * score.duration.quarterLength)
            data = np.zeros((4, num_frames, d))

            for n in range(num_frames):
                t_ql = float(n) / frames_per_crotchet

                all_voices_context_idx = np.zeros((2*4,all_voices_context_size)) # 4 voices, 2 events (played,articulated)
                # NOTE: fragile (0,1) indexing to retrieve previous note and articulation
                all_voices_context_idx[:4,:] = np.pad(
                        data[:, max(0,(n-all_voices_context_size)):n, 0],
                        ((0,0), (max(all_voices_context_size - n, 0), 0)),
                        mode='constant')
                all_voices_context_idx[4:,:] = np.pad(
                        data[:, max(0,(n-all_voices_context_size)):n, 1],
                        ((0,0), (max(all_voices_context_size - n, 0), 0)),
                        mode='constant')

                for part in score.parts:
                    part_idx = part_id_to_idx[part.id]
                    part_context_idx = np.pad(
                            data[part_idx, max(0,(n-part_context_size)):n, 2], # NOTE: fragile way to get pitch class, will change if columns in `data` changes
                            (max(part_context_size  - n, 0),0),
                            mode='constant')
                    measure = (part\
                        .getElementsByClass('Measure')\
                        .getElementsByOffset(t_ql, mustBeginInSpan=False, includeElementsThatEndAtStart=False))[0]
                    nr = measure.notesAndRests.getElementsByOffset(t_ql % 4, mustBeginInSpan=False, includeElementsThatEndAtStart=False)
                    if nr:
                        assert len(nr) == 1
                        nr = nr[0]
                        if nr.isRest:
                            note_name = 'REST'
                        elif nr.isChord:
                            note_name = nr.findRoot().nameWithOctave
                        else:
                            note_name = nr.nameWithOctave

                        # we +1 because '0' is used as 'mask' in `Embedding` for note, pc, and beat
                        note_idx = note_name_to_idx[note_name] + 1
                        if nr.isNote:
                            f = nr.pitch.frequency
                            pc = nr.pitch.pitchClass + 1
                        else:
                            f = 0
                            pc = 1
                        beat = int(2*nr.getOffsetBySite(measure)) + 1

                        new_notes = measure.notesAndRests.getElementsByOffset(t_ql % 4, mustBeginInSpan=True, includeElementsThatEndAtStart=False)
                        articulated = nr in new_notes

                        data[part_idx, n] = np.hstack((
                            np.array([note_idx, articulated, pc, f, beat]),
                            part_context_idx,
                            np.reshape(all_voices_context_idx, -1)
                            ))
            X_all[score_idx,:,:data.shape[1],] = data

        np.save(fp, X_all)
        return len(unique_notes), X_all

def _ohe(index, size):
    x = np.zeros((size,))
    x[index] = 1
    return x

map(keras.add_command, [
    train_lstm,
    sample_lstm,
    train_discrim,
    biaxial
])


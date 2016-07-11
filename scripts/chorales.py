import click
import cPickle
import json
import multiprocess as mp

from music21 import analysis, converter, corpus, meter
from music21.note import Note

from constants import *

@click.group()
def chorales():
    """Constructs various corpuses using BWV Bach chorales."""
    pass

def prepare_standard(subset):
    """Prepare scores by standardizing names and transposing to Cmaj/Amin"""
    dataset = list()
    it = corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream')
    if subset:
        it = [next(it) for _ in range(5)]
    for sc in it:
        bwv_id = sc.metadata.title
        sc = standardize_part_ids(sc)
        if sc:
            print 'Processing ' + bwv_id
            dataset.append(sc)
        else:
            print 'Skipping ' + bwv_id + ', error extracting parts'
    return dataset

@click.command()
def prepare_mono_all_constant_t():
    """Prepares all monophonic parts, constant timestep between samples.

        * Start notes are prefixed with a special `NOTE_START_SYM`
        * Each quarter note is expanded to `FRAMES_PER_CROTCHET` frames
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = standardize_key(score)
            key = score.analyze('key')
            for part in score.parts:
                note_duration_pairs = list(_encode_note_duration_tuples(part))

                assert all(map(lambda x: x >= 1.0, set([FRAMES_PER_CROTCHET * dur for _,dur in note_duration_pairs]))),\
                        "Could not quantize constant timesteps"

                pairs_text = []
                for note,dur in note_duration_pairs:
                    pairs_text.append(NOTE_START_SYM + note)
                    for _ in range(1,int(FRAMES_PER_CROTCHET*dur)):
                        pairs_text.append(note)
                yield ('{0}-{1}-{2}-mono-all'.format(bwv_id, key.mode, part.id), pairs_text)
    _process_scores_with(_fn)

@click.command()
@click.option('--soprano-only',
        type=bool, default=False, help='Only extract Soprano parts')
@click.option('--use-pitch-classes',
        type=bool, default=False, help='Use pitch equivalence classes, discarding octave information')
def prepare_mono_all(soprano_only, use_pitch_classes):
    """Prepares a corpus containing all monophonic parts, with major/minor labels.

        * Only 4/4 time signatures are considered
        * The key is transposed to Cmaj/Amin
        * All monophonic parts are extracted and sequentially concatenated
        * Multiple (Pitch,Duration) sequence are returned
        * The files output have names `{bwv_id}-{major|minor}-{part_id}`
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = standardize_key(score)
            key = score.analyze('key')
            parts = []
            if soprano_only:
                parts.append(_get_soprano_part(score))
            else:
                parts = score.parts
            for part in score.parts:
                note_duration_pairs = _encode_note_duration_tuples(part)
                if use_pitch_classes:
                    note_duration_pairs = map(
                            lambda x: x if (x[0] == u'REST') else (Note(x[0]).name, x[1]),
                            note_duration_pairs)
                pairs_text = map(lambda entry: '{0},{1}'.format(*entry), note_duration_pairs)
                if soprano_only:
                    yield ('{0}-{1}-soprano-mono'.format(bwv_id, key.mode, part.id), pairs_text)
                else:
                    yield ('{0}-{1}-{2}-mono'.format(bwv_id, key.mode, part.id), pairs_text)
    _process_scores_with(_fn)

@click.command()
def prepare_durations():
    """Prepares a corpus containing durations from all parts."""
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = standardize_key(score)
            key = score.analyze('key')
            for part in score.parts:
                durations = list(map(lambda note: note.quarterLength, part))
                text = map(lambda entry: '{0}'.format(entry), durations)
                yield ('{0}-{1}-{2}-duration'.format(bwv_id, key.mode, part.id), text)
    _process_scores_with(_fn)

@click.command()
def prepare_poly():
    """
    Prepares a corpus of all four parts.

    Each score is transformed into a sequence of tuples with a constant
    timestep of (1/`FRAMES_PER_CROTCHET`) crotchets between consecutive chords.

    Each encoded chord has the following format:
        Notes : List[(
            Midi: Int,
            Tied : Bool (true if note is continuation of previous note)
        )]
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = standardize_key(score)
            key = score.analyze('key')

            encoded_score = []
            for chord in score.chordify().flat.notesAndRests: # aggregate voices, remove markup
                # expand chord/rest s.t. constant timestep between frames

                # TODO: handle rest
                if chord.isRest:
                    encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET)) * [[]])
                else:
                    # add ties with previous chord if present
                    encoded_score.append(map(
                        lambda note: (note.pitch.midi, note.tie is not None and note.tie.type != 'start'),
                        chord))

                    # repeat pitches to expand chord into multiple frames
                    # all repeated frames when expanding a chord should be tied
                    encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET) - 1) * [map(
                        lambda note: (note.pitch.midi, True),
                        chord)])

            yield ('beethoven-{0}-{1}-chord-constant-t'.format(bwv_id, key.mode), encoded_score)

    plain_text_data = []

    # construct vocab <=> UTF8 mapping
    vocabulary = set([str((midi, tie)) for tie in [True, False] for midi in range(128)]) # all MIDI notes and tie/notie
    vocabulary.add(CHORD_BOUNDARY_DELIM)

    pairs_to_utf = dict(map(lambda x: (x[1], unichr(x[0])), enumerate(vocabulary)))
    utf_to_txt = {utf:txt for txt,utf in pairs_to_utf.items()}

    # score start/end delimiters are added during concatenation
    utf_to_txt[START_DELIM] = 'START'
    utf_to_txt[END_DELIM] = 'END'

    p = mp.Pool(processes=mp.cpu_count())
    # processed_scores = map(lambda score: list(_fn(score)), corpus.chorales.Iterator(
    #     numberingSystem='bwv',
    #     returnType='stream'))

    processed_scores = map(lambda score: list(_fn(corpus.parse(score))), corpus.getComposer('beethoven','xml'))

    for processed_score in processed_scores:
        for fname, encoded_score in processed_score:
            encoded_score_plaintext = []
            for i,chord in enumerate(encoded_score):
                if i > 0: encoded_score_plaintext.append(CHORD_BOUNDARY_DELIM) # chord boundary delimiter
                for note in chord:
                    encoded_score_plaintext.append(str(note))
            plain_text_data.append((fname, encoded_score_plaintext))

    # save outputs
    with open(SCRATCH_DIR + '/utf_to_txt.json', 'w') as fd:
        print 'Writing ' + SCRATCH_DIR + '/utf_to_txt.json'
        json.dump(utf_to_txt, fd)

    for fname, plain_text in plain_text_data:
        out_path = SCRATCH_DIR + '/{0}'.format(fname)
        print 'Writing {0}'.format(out_path)
        with open(out_path + '.txt', 'w') as fd:
            fd.write('\n'.join(plain_text))
        with open(out_path + '.utf', 'w') as fd:
            fd.write('\n'.join(map(pairs_to_utf.get, plain_text)))

def _process_scores_with(fn):
    """Extracts data from all BWV scores using `fn`.

    `fn` should take a `music21.stream.Score` and return a `[(FileName, [String]|None)]` where
    each element represents an extracted univariate sequence of discrete tokens from the
    score.

        * `music21` is used to get Bach chorales using BWV numbering system
        * Each chorale is processed using `fn`
        * The output is written to `${SCRATCH_DIR}/${FileName}.{txt,utf}
        * `utf_to_txt.json` is a dictionary mapping UTF8 symbols to plain text

    Existing files are overwritten because the vocabulary can change between runs.
    """
    # used for encoding/decoding tokens to UTF8 symbols
    plain_text_data = []
    vocabulary = set() # remember all unique (note,duration) tuples seen

    p = mp.Pool(processes=mp.cpu_count())
    processed_scores = p.map(lambda score: list(fn(score)), corpus.chorales.Iterator(
            numberingSystem='bwv',
            returnType='stream'))
    for processed_score in processed_scores:
        for fname, pairs_text in processed_score:
            if pairs_text:
                plain_text_data.append((fname, pairs_text))
                vocabulary.update(set(pairs_text))

    # construct vocab <=> UTF8 mapping
    pairs_to_utf = dict(map(lambda x: (x[1], unichr(x[0])), enumerate(vocabulary)))
    utf_to_txt = {utf:txt for txt,utf in pairs_to_utf.items()}
    utf_to_txt[START_DELIM] = 'START'
    utf_to_txt[END_DELIM] = 'END'

    # save outputs
    with open(SCRATCH_DIR + '/utf_to_txt.json', 'w') as fd:
        print 'Writing ' + SCRATCH_DIR + '/utf_to_txt.json'
        json.dump(utf_to_txt, fd)

    for fname, pairs_text in plain_text_data:
        out_path = SCRATCH_DIR + '/{0}'.format(fname)
        print 'Writing {0}'.format(out_path)
        with open(out_path + '.txt', 'w') as fd:
            fd.write('\n'.join(pairs_text))
        with open(out_path + '.utf', 'w') as fd:
            fd.write('\n'.join(map(pairs_to_utf.get, pairs_text)))

def standardize_part_ids(bwv_score):
    "Standardizes the `id`s of `parts` (Soprano, Alto, etc) from `corpus.chorales.Iterator(numberingSystem='bwv')`"
    ids = dict()
    ids['Soprano'] = {
            'Soprano',
            'S.',
            'Soprano 1', # NOTE: soprano1 or soprano2?
            'Soprano\rOboe 1\rViolin1'}
    ids['Alto'] = { 'Alto', 'A.'}
    ids['Tenor'] = { 'Tenor', 'T.'}
    ids['Bass'] = { 'Bass', 'B.'}
    id_to_name = {id:name for name in ids for id in ids[name] }
    all_ids = set(id_to_name.keys())
    if all(map(lambda part: part.id in all_ids, bwv_score.parts)):
        for part in bwv_score.parts:
            part.id = id_to_name[part.id]
        return bwv_score
    else:
        return None

def _get_part(bwv_score, part_ids):
    """Tries to extract part matching names in `part_ids`."""
    is_match = map(lambda part: part.id in part_ids, bwv_score.parts)
    if sum(is_match) == 1:
        return bwv_score.parts[is_match.index(True)]
    else:
        return None


def standardize_key(score):
    """Converts into the key of C major or A minor.

    Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
    """
    # major conversions
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("F#",6),("G-", 6),("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#",3),("G-", 3),("G", 2)])

    # transpose score
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    tScore = score.transpose(halfSteps)

    # transpose key signature
    for ks in tScore.flat.getKeySignatures():
        ks.transpose(halfSteps, inPlace=True)
    return tScore


def _encode_note_duration_tuples(part):
    """
    Generator yielding notes/rests and durations (in crotchets) for a part.

    Notes are encoded with their MIDI value and rests are encoded as -1.
    """
    for nr in part.flat.notesAndRests:
        if nr.isNote:
            yield (nr.midi, nr.quarterLength)
        else:
            yield (-1, nr.quarterLength)

map(chorales.add_command, [
    prepare_mono_all,
    prepare_durations,
    prepare_mono_all_constant_t,
    prepare_poly
])

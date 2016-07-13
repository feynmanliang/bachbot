import click
import cPickle
import json
import multiprocess as mp

from music21 import analysis, converter, corpus, meter
from music21.note import Note

from constants import *

@click.group()
def dataset_utils():
    """Constructs various datasets."""
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
def prepare_chorales_poly():
    """
    Prepares polyphonic scores using a chord tuple representation.

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
    processed_scores = p.map(lambda score: list(_fn(score)), corpus.chorales.Iterator(
        numberingSystem='bwv',
        returnType='stream'))

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

map(dataset_utils.add_command, [
    prepare_chorales_poly
])

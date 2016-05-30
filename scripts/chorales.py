import click
import json
import multiprocess as mp

from music21 import analysis, converter, corpus, meter

from constants import *

@click.group()
def chorales():
    """Constructs various corpuses using BWV Bach chorales."""
    pass

@click.command()
def prepare_mono_all_constant_t():
    """Prepares all monophonic parts, constant timestep between samplesi.

        * Start notes are prefixed with a special `NOTE_START_SYM`
        * Each quarter note is expanded to `FRAMES_PER_CROTCHET` frames
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = _standardize_key(score)
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
def prepare_mono_all():
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

            score = _standardize_key(score)
            key = score.analyze('key')
            for part in score.parts:
                note_duration_pairs = list(_encode_note_duration_tuples(part))
                pairs_text = map(lambda entry: '{0},{1}'.format(*entry), note_duration_pairs)
                yield ('{0}-{1}-{2}-mono-all'.format(bwv_id, key.mode, part.id), pairs_text)
    _process_scores_with(_fn)

@click.command()
def prepare_soprano():
    """Prepares a corpus containing all soprano parts.

        * Only 4/4 time signatures are considered
        * The key is transposed to Cmaj/Amin
        * Only the soprano part is extracted
        * A (Pitch,Duration) sequence is returned
        * The files output have names `{bwv_id}-soprano`
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = _standardize_key(score)
            soprano_part = _get_soprano_part(score)
            note_duration_pairs = list(_encode_note_duration_tuples(soprano_part))
            pairs_text = map(lambda entry: '{0},{1}'.format(*entry), note_duration_pairs)
            yield ('{0}-soprano'.format(bwv_id), pairs_text)
    _process_scores_with(_fn)

@click.command()
def prepare_durations():
    """Prepares a corpus containing durations from all parts."""
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = _standardize_key(score)
            key = score.analyze('key')
            for part in score.parts:
                note_duration_pairs = list(map(lambda note: note.quarterLength, part))
                pairs_text = map(lambda entry: '{0}'.format(entry), note_duration_pairs)
                yield ('{0}-{1}-{2}-duration'.format(bwv_id, key.mode, part.id), pairs_text)
    _process_scores_with(_fn)

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

def _get_soprano_part(bwv_score):
    """Extracts soprano line from `corpus.chorales.Iterator(numberingSystem='bwv')` elements."""
    soprano_part_ids = set([
            'Soprano',
            'S.',
            'Soprano 1', # TODO: soprano1 or soprano2
            'Soprano\rOboe 1\rViolin1'])
    bwv_to_soprano_id = { # NOTE: these appear to all be score.parts[0]
            '277': 'spine_6',
            '281': 'spine_3',
            '366': 'spine_8'
            }
    bwv_id = bwv_score.metadata.title
    if bwv_id in bwv_to_soprano_id:
        return bwv_score.parts[bwv_to_soprano_id[bwv_id]]
    else:
        is_soprano_part = map(lambda part: part.id in soprano_part_ids, bwv_score.parts)
        assert sum(is_soprano_part) == 1, \
                'Could not find a unique soprano part id from: {0}'.format(
                        map(lambda part: part.id, bwv_score.parts))
        return bwv_score.parts[is_soprano_part.index(True)]


def _standardize_key(score):
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
    """Generator yielding the notes/rests and durations for a single part."""
    for nr in part.flat.notesAndRests:
        if nr.isNote:
            yield (nr.nameWithOctave, nr.quarterLength)
        else:
            yield ('REST',nr.quarterLength)

map(chorales.add_command, [
    prepare_soprano,
    prepare_mono_all,
    prepare_durations,
    prepare_mono_all_constant_t
])

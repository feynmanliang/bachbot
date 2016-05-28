import click
import json

from music21 import analysis, converter, corpus, meter

from constants import *

@click.command()
def prepare_bach_chorales_mono():
    """Prepares the Bach chorale training dataset.

    Existing files are overwritten because the vocabulary can change between runs.

    The following steps are performed:
        * `music21` is used to get Bach chorales using BWV numbering system
        * Only 4/4 time signatures are considered
        * The soprano / top part is extraced
        * The data is encoded into a `(Note+Octave|Rest, Duration) :: (Char+Int|'REST', Float)` format
          and written to `bachbot/scratch/{bwv_id}-mono.{txt,utf,json}`
            * The `.txt` file contains the tuples in plain text
            * The `.utf` file contains the encoding of the tuples into unique UTF8 symbols
            * The `.json` file contains a dictionary mapping UTF8 symbols to decoded tuples
    """
    for score in corpus.chorales.Iterator(
            numberingSystem='bwv',
            returnType='stream'):
            #analysis=True): # analysis only available for riemenschneider

        # used for UTF8 encoding later
        plain_text_data = []
        vocabulary = set() # remember all unique (note,duration) tuples seen

        # convert all the files and build vocabulary
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            out_path = SCRATCH_DIR + '/{0}-mono'.format(bwv_id)

            score = _standardize_key(score)
            soprano_part = _get_soprano_part(score)
            note_duration_pairs = list(_encode_note_duration_tuples(soprano_part))
            pairs_text = map(lambda entry: '{0},{1}'.format(*entry), note_duration_pairs)
            plain_text_data.append((out_path, pairs_text))
            for txt in pairs_text:
                vocabulary.add(txt)

        # construct vocab <=> UTF8 mapping
        pairs_to_utf = dict(map(lambda x: (x[1], unichr(x[0])), enumerate(vocabulary)))

        # save outputs
        for out_path, pairs_text in plain_text_data:
            with open(out_path + '.txt', 'w') as fd:
                fd.write('\n'.join(pairs_text))
            with open(out_path + '.utf', 'w') as fd:
                fd.write('\n'.join(map(pairs_to_utf.get, pairs_text)))
            with open(out_path + '.json', 'w') as fd:
                utf_to_txt = {utf:txt for txt,utf in pairs_to_utf.items()}
                json.dump(utf_to_txt, fd)

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

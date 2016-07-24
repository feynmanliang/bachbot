import click

import json, cPickle
import requests, zipfile

import os, glob

from music21 import analysis, converter, corpus, meter
from music21.note import Note

from constants import *

@click.group()
def datasets():
    """Constructs various datasets."""
    pass

@click.command()
@click.option('--ignore-fermatas', type=bool, default=False)
@click.option('--subset', type=bool, default=True)
def prepare_poly(ignore_fermatas, subset):
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
    txt_to_utf, utf_to_txt = build_vocabulary()

    it = corpus.chorales.Iterator(
        numberingSystem='bwv',
        returnType='stream')
    if subset:
        it = [next(it) for _ in range(5)]

    for score in it:
        if score.getTimeSignatures()[0].ratioString != '4/4': # only consider 4/4
            continue

        bwv_id = score.metadata.title
        print('Processing BWV {0}'.format(bwv_id))

        score = standardize_key(score)
        key = score.analyze('key')
        encoded_score = encode_score(score, ignore_fermatas=ignore_fermatas)

        fname = 'BWV-{0}-{1}'.format(bwv_id, key.mode)
        if not ignore_fermatas:
            fname += '-fermatas'

        encoded_score_txt = to_text(encoded_score)

        out_path = SCRATCH_DIR + '/{0}'.format(fname)
        print 'Writing {0}'.format(out_path)
        with open(out_path + '.txt', 'w') as fd:
            fd.write('\n'.join(encoded_score_txt))
        with open(out_path + '.utf', 'w') as fd:
            fd.write('\n'.join(map(txt_to_utf.get, encoded_score_txt)))

def encode_score(score, ignore_fermatas=False):
    """
    Encodes a music21 score into a List of chords, where each chord is represented with
    a (Fermata :: Bool, List[(Note :: Integer, Tie :: Bool)]).

    If `ignore_fermatas` is True, all `has_fermata`s will be False.

    Time is discretized such that each crotchet occupies `FRAMES_PER_CROTCHET` frames.
    """
    encoded_score = []
    for chord in score.chordify().flat.notesAndRests: # aggregate voices, remove markup
        # expand chord/rest s.t. constant timestep between frames

        # TODO: handle rest
        if chord.isRest:
            encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET)) * [[]])
        else:
            has_fermata = False
            if not ignore_fermatas:
                has_fermata = any(map(lambda e: e.isClassOrSubclass(('Fermata',)), chord.expressions))

            # add ties with previous chord if present
            encoded_score.append((has_fermata, map(
                lambda note: (note.pitch.midi, note.tie is not None and note.tie.type != 'start'),
                chord)))

            # repeat pitches to expand chord into multiple frames
            # all repeated frames when expanding a chord should be tied
            encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET) - 1) * [
                (has_fermata,
                    map(lambda note: (note.pitch.midi, True), chord))
            ])
    return encoded_score

@click.command()
@click.option('--mask-part', '-m', multiple=True, help='Parts (Soprano, Alto, Tenor, Bass) to mask')
def prepare_harm(mask_part):
    """
    Prepares harmonization data.
    """
    def _fn(score):
        if score.getTimeSignatures()[0].ratioString == '4/4': # only consider 4/4
            bwv_id = score.metadata.title
            print('Processing BWV {0}'.format(bwv_id))

            score = standardize_key(score)
            score = extract_SATB(score)
            key = score.analyze('key')

            encoded_score = encode_score(score, mask_part)

            yield ('BWV-{0}-{1}-mask-{2}'.format(bwv_id, key.mode, '-'.join(mask_part)), encoded_score)

    def encode_score(score, parts_to_mask):
        encoded_score = []
        for chord in score.chordify(addPartIdAsGroup=True).flat.notesAndRests: # aggregate voices, remove markup
            # expand chord/rest s.t. constant timestep between frames
            has_fermata = any(map(lambda e: e.isClassOrSubclass(('Fermata',)), chord.expressions))
            if chord.isRest:
                encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET)) * [[]])
            else:
                # add ties with previous chord if present
                encoded_chord = []
                for note in chord: # TODO: sort ascending on both training and eval
                    if note.pitch.groups[0] in parts_to_mask:
                        encoded_chord.append(BLANK_MASK_TXT)
                    else:
                        has_tie = note.tie is not None and note.tie.type != 'start'
                        encoded_chord.append((note.pitch.midi, has_tie))
                encoded_score.append((has_fermata, encoded_chord))

                # repeat pitches to expand chord into multiple frames
                # all repeated frames when expanding a chord should be tied
                encoded_score.extend((int(chord.quarterLength * FRAMES_PER_CROTCHET) - 1) * [
                    (has_fermata,
                        map(lambda note: BLANK_MASK_TXT if note == BLANK_MASK_TXT else (note[0], True), encoded_chord))
                ])
        return encoded_score

    utf_to_txt = json.load(open(SCRATCH_DIR + '/utf_to_txt.json', 'rb'))
    txt_to_utf = { v:k for k,v in utf_to_txt.items() }
    txt_to_utf[BLANK_MASK_TXT] = BLANK_MASK_UTF

    it = corpus.chorales.Iterator(
        numberingSystem='bwv',
        returnType='stream')
    scores = [next(it) for _ in range(5)]

    processed_scores = map(lambda score: list(_fn(score)), scores)

    plain_text_data = []
    for processed_score in processed_scores:
        for fname, encoded_score in processed_score:
            encoded_score_plaintext = to_text(encoded_score)
            plain_text_data.append((fname, encoded_score_plaintext))

    for fname, plain_text in plain_text_data:
        out_path = SCRATCH_DIR + '/harm/{0}'.format(fname)
        print 'Writing {0}'.format(out_path)
        with open(out_path + '.txt', 'w') as fd:
            fd.write('\n'.join(plain_text))
        with open(out_path + '.utf', 'w') as fd:
            fd.write(_encode_text(txt_to_utf, encoded_score_plaintext))

@click.command()
@click.option('--utf-to-txt-json', type=click.File('rb'), default=SCRATCH_DIR + '/utf_to_txt.json')
@click.argument('in-file', type=click.File('rb'))
@click.argument('out-file', type=click.File('wb'))
def encode_text(utf_to_txt_json, in_file, out_file):
    utf_to_txt = json.load(utf_to_txt_json)
    txt_to_utf = { v:k for k,v in utf_to_txt.items() }
    out_file.write(_encode_text(txt_to_utf, in_file))

def _encode_text(txt_to_utf, score_txt):
    """
    Converts a text-encoded score into UTF encoding (appending start/end delimiters).

    Throws `KeyError` when out-of-vocabulary token is encountered
    """
    return START_DELIM +\
            ''.join(map(lambda txt: txt_to_utf[txt.strip()], score_txt)) +\
            END_DELIM

def standardize_key(score):
    """Converts into the key of C major or A minor.

    Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
    """
    # major conversions
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("C#",-1),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("F#",6),("G-", 6),("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("C#",-4),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#",3),("G-", 3),("G", 2)])

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

def extract_SATB(score):
    """
    Extracts the Soprano, Alto, Tenor, and Bass parts from a piece. The returned score is guaranteed
    to have parts with names 'Soprano', 'Alto', 'Tenor', and 'Bass'.

    This method mutates its arguments.
    """
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
    for part in score.parts:
        if part.id in id_to_name:
            part.id = id_to_name[part.id]
        else:
            score.remove(part)
    return score

def build_vocabulary():
    vocabulary = set([str((midi, tie)) for tie in [True, False] for midi in range(128)]) # all MIDI notes and tie/notie
    vocabulary.update(set([CHORD_BOUNDARY_DELIM, FERMATA_SYM]))
    txt_to_utf = dict(map(lambda x: (x[1], unichr(x[0])), enumerate(vocabulary)))
    txt_to_utf['START'] = START_DELIM
    txt_to_utf['END'] = END_DELIM
    utf_to_txt = {utf:txt for txt,utf in txt_to_utf.items()}
    # save vocabulary
    with open(SCRATCH_DIR + '/utf_to_txt.json', 'w') as fd:
        print 'Writing vocabulary to ' + SCRATCH_DIR + '/utf_to_txt.json'
        json.dump(utf_to_txt, fd)
    return txt_to_utf, utf_to_txt

def to_text(encoded_score):
    "Converts a Python encoded score into plain-text."
    encoded_score_plaintext = []
    for i,chord_pair in enumerate(encoded_score):
        if i > 0:
            encoded_score_plaintext.append(CHORD_BOUNDARY_DELIM) # chord boundary delimiter
        if len(chord_pair) > 0:
            is_fermata, chord = chord_pair
            if is_fermata:
                encoded_score_plaintext.append(FERMATA_SYM)
            for note in chord:
                encoded_score_plaintext.append(str(note))
    return encoded_score_plaintext

map(datasets.add_command, [
    prepare_poly,
    prepare_harm,
    encode_text,
])

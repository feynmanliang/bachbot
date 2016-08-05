import click

from constants import *
from datasets import to_text

from music21.stream import Stream
from music21.note import Note, Rest
from music21.tie import Tie
from music21.duration import Duration
from music21.chord import Chord
from music21 import expressions

import codecs
import json

@click.group()
def decode():
    "Decode encoded data format into musicXML for displaying and playback."
    pass

@click.command()
@click.option('--utf-to-txt-json', type=click.File('rb'), default=SCRATCH_DIR + '/utf_to_txt.json')
@click.option('--min-length', type=int, default=100, help='Minimum length of samples in frames')
@click.option('--max-length', type=int, default=400, help='Maximum length of samples in frames')
@click.argument('utf8-file', type=click.Path(exists=True))
@click.argument('out-dir', type=click.Path(exists=True), default=SCRATCH_DIR + '/out')
def sampled_stream(utf_to_txt_json, utf8_file, out_dir, min_length, max_length):
    """
    Decodes all scores in a single sampled UTF stream to text and musicXML.

    This method splits on START_DELIM and outputs one text and musicXML encoded file for each score.
    """
    utf_to_txt = json.load(utf_to_txt_json)
    utf_data = filter(lambda x: x != u'\n', codecs.open(utf8_file, "r", "utf-8").read())
    utf_scores = utf_data.split(START_DELIM)[1:] # [1:] ignores first START_DELIM

    i = 0
    for utf_score in utf_scores:
        score = decode_utf_single(utf_to_txt, utf_score)
        if score and len(score) >= min_length and len(score) <= max_length:
            print('Writing {0}'.format(out_dir + '/out-{0}'.format(i)))
            with open(out_dir + '/out-{0}.txt'.format(i), 'w') as fd:
                fd.write('\n'.join(to_text(score)))
            to_musicxml(score).write('musicxml', out_dir + '/out-{0}.xml'.format(i))
            i += 1

@click.command()
@click.option('--utf-to-txt-json', type=click.File('rb'), default=SCRATCH_DIR + '/utf_to_txt.json')
@click.argument('utf8-file', type=click.Path(exists=True))
@click.argument('out-file', type=click.File('wb'), default=SCRATCH_DIR + '/out/decode.xml')
def single(utf_to_txt_json, utf8_file, out_file):
    """
    Decodes a single UTF8 output file
    """
    utf_to_txt = json.load(utf_to_txt_json)
    utf_data = filter(lambda x: x != u'\n', codecs.open(utf8_file, "r", "utf-8").read())
    utf_scores = utf_data.split(START_DELIM)[1:] # [1:] ignores first START_DELIM

    score = decode_utf_single(utf_to_txt, utf_scores[0])
    print('Writing {0}'.format(out_file.name))
    if score:
        to_musicxml(score).write('musicxml', out_file.name)

def decode_utf_single(utf_to_txt, utf_score):
    "Reads a single UTF encoded file into a Python representation."
    curr_score = []
    curr_chord_fermata = False
    curr_chord_notes = []
    i = 0
    for utf_token in utf_score:
        txt = utf_to_txt.get(utf_token)
        if txt == 'END':
            return curr_score
        elif txt == 'START':
            curr_score = []
            curr_chord_notes = []
        elif txt == CHORD_BOUNDARY_DELIM:
            curr_score.append((curr_chord_fermata, curr_chord_notes))
            curr_chord_fermata = False
            curr_chord_notes = []
        elif txt == FERMATA_SYM:
            curr_chord_fermata = True
        elif txt == None:
            print(u'Skipping unknown token: {}'.format(utf_token))
        else:
            curr_chord_notes.append(eval(txt))

def to_musicxml(sc_enc):
    "Converts Chord tuples (see chorales.prepare_poly) to musicXML"
    timestep = Duration(1. / FRAMES_PER_CROTCHET)
    musicxml_score = Stream()
    prev_chord = dict() # midi->(note instance from previous chord), used to determine tie type (start, continue, stop)
    for has_fermata, chord_notes in sc_enc:
        notes = []
        if len(chord_notes) == 0: # no notes => rest for this frame
            r = Rest()
            r.duration = timestep
            musicxml_score.append(r)
        else:
            for note_tuple in chord_notes:
                note = Note()
                if has_fermata:
                    note.expressions.append(expressions.Fermata())
                note.midi = note_tuple[0]
                if note_tuple[1]: # current note is tied
                    note.tie = Tie('stop')
                    if prev_chord and note.pitch.midi in prev_chord:
                        prev_note = prev_chord[note.pitch.midi]
                        if prev_note.tie is None:
                            prev_note.tie = Tie('start')
                        else:
                            prev_note.tie = Tie('continue')
                notes.append(note)
            prev_chord = { note.pitch.midi : note for note in notes }
            chord = Chord(notes=notes, duration=timestep)
            if has_fermata:
                chord.expressions.append(expressions.Fermata())
            musicxml_score.append(chord)
    return musicxml_score

map(decode.add_command, [
    sampled_stream,
    single,
])

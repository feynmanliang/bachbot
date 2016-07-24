import click

from constants import *

from music21.stream import Stream
from music21.note import Note
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
@click.argument('json-file', type=click.File('rb'))
@click.argument('utf8-file', type=click.Path(exists=True))
def decode_utf(json_file, utf8_file):
    "Decodes UTF string encoded score into musicXML output."
    out_dir = SCRATCH_DIR + '/out'

    utf_to_txt = json.load(json_file)
    utf8_file = codecs.open(utf8_file, "r", "utf-8")

    curr_file = []
    curr_chord_notes = []
    i = 0
    for txt in map(utf_to_txt.get, filter(lambda x: x != u'\n', utf8_file.read())):
        if txt == 'START':
            curr_file = []
        elif txt == 'END':
            if not os.path.exists(out_dir):
                print('Creating directory {0}'.format(out_dir))
                os.makedirs(out_dir)
            print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
            to_musicxml(curr_file).write('musicxml', out_dir + '/out-{0}.xml'.format(i))
            i += 1
        elif txt == CHORD_BOUNDARY_DELIM:
            curr_file.append(curr_chord_notes)
            curr_chord_notes = []
        else:
            curr_chord_notes.append(eval(txt))
    print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
    to_musicxml(curr_file).write('musicxml', out_dir + '/out-{0}.xml'.format(i))

def to_musicxml(sc_enc):
    "Converts Chord tuples (see chorales.prepare_poly) to musicXML"
    timestep = Duration(1. / FRAMES_PER_CROTCHET)
    musicxml_score = Stream()
    prev_chord = dict() # midi->(note instance from previous chord), used to determine tie type (start, continue, stop)
    for chord_notes in sc_enc:
        notes = []
        for note_tuple in chord_notes:
            # TODO: handle rests
            note = Note()
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
        musicxml_score.append(Chord(notes=notes, duration=timestep))
    return musicxml_score

@click.command()
@click.argument('json-file', type=click.File('rb'))
@click.argument('utf8-file', type=click.Path(exists=True))
def decode_utf_fermata(json_file, utf8_file):
    "Decodes UTF string encoded score (with fermatas) into musicXML output."
    out_dir = SCRATCH_DIR + '/out'

    utf_to_txt = json.load(json_file)
    utf8_file = codecs.open(utf8_file, "r", "utf-8")

    curr_file = []
    curr_chord_fermata = False
    curr_chord_notes = []
    i = 0
    for txt in map(utf_to_txt.get, filter(lambda x: x != u'\n', utf8_file.read())):
        print txt
        if txt == 'START':
            curr_file = []
        elif txt == 'END':
            if not os.path.exists(out_dir):
                print('Creating directory {0}'.format(out_dir))
                os.makedirs(out_dir)
            print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
            to_musicxml_fermata(curr_file).write('musicxml', out_dir + '/out-{0}.xml'.format(i))
            i += 1
        elif txt == CHORD_BOUNDARY_DELIM:
            curr_file.append((curr_chord_fermata, curr_chord_notes))
            curr_chord_fermata = False
            curr_chord_notes = []
        elif txt == FERMATA_SYM:
            curr_chord_fermata = True
        else:
            curr_chord_notes.append(eval(txt))
    print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
    to_musicxml_fermata(curr_file).write('musicxml', out_dir + '/out-{0}.xml'.format(i))

def to_musicxml_fermata(sc_enc):
    "Converts Chord tuples (see chorales.prepare_poly) to musicXML"
    timestep = Duration(1. / FRAMES_PER_CROTCHET)
    musicxml_score = Stream()
    prev_chord = dict() # midi->(note instance from previous chord), used to determine tie type (start, continue, stop)
    for has_fermata, chord_notes in sc_enc:
        notes = []
        for note_tuple in chord_notes:
            # TODO: handle rests
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
    decode_utf,
    decode_utf_fermata,
])

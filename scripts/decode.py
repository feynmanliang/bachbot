import click

from constants import *

from music21.stream import Stream
from music21.note import Note
from music21.tie import Tie
from music21.duration import Duration
from music21.chord import Chord

import codecs
import json

@click.group()
def decode():
    "Decode encoded data format into musicXML for displaying and playback."
    pass

@click.command()
@click.argument('json-file', type=click.File('rb'))
@click.argument('utf8-file', type=click.Path(exists=True))
def decode_chord_constant_t_utf(json_file, utf8_file):
    "Decodes plain text encoding made by `chorales.prepare_poly` into python tuples."
    utf_to_txt = json.load(json_file)
    utf8_file = codecs.open(utf8_file, "r", "utf-8")

    sc_enc = []
    curr_chord_notes = []
    for txt in map(utf_to_txt.get, filter(lambda x: x != u'\n', utf8_file.read())):
        if txt == CHORD_BOUNDARY_DELIM:
            sc_enc.append(curr_chord_notes)
            curr_chord_notes = []
        else:
            curr_chord_notes.append(eval(txt))

    to_musicxml(sc_enc).write(fp='out/out.xml')

def to_musicxml(sc_enc):
    "Converts Chord tuples (see chorales.prepare_poly) to musicXML"
    timestep = Duration(1. / FRAMES_PER_CROTCHET)
    musicxml_score = Stream()
    prev_chord = dict() # midi->(note instance from previous chord), used to determine tie type (start, continue, stop)
    for chord_notes in sc_enc:
        notes = []
        for note_tuple in chord_notes:
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



map(decode.add_command, [
    decode_chord_constant_t_utf
])

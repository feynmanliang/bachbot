import click

from constants import *

from music21.stream import Stream
from music21.note import Note
from music21.tie import Tie
from music21.duration import Duration
from music21.chord import Chord

@click.group()
def decode():
    "Decode encoded data format into musicXML for displaying and playback."
    pass

#@click.command()
def decode_chord_constant_t(sc_enc):
    "Decodes encoding made by chorales.prepare_poly."
    timestep = Duration(1. / FRAMES_PER_CROTCHET)
    decoded_score = Stream()
    prev_chord = dict() # midi->(note instance from previous chord), used to determine tie type (start, continue, stop)
    for chord_notes in sc_enc:
        notes = []
        for note_tuple in chord_notes:
            note = Note()
            note.midi = note_tuple[0]
            if note_tuple[1]: # current note is tied
                note.tie = Tie('stop')
                if prev_chord and note.midi in prev_chord:
                    prev_note = prev_chord[note.midi]
                    if prev_note.tie is None:
                        prev_note.tie = Tie('start')
                    else:
                        prev_note.tie = Tie('continue')
            notes.append(note)
        prev_chord = { note.midi : note for note in notes }
        decoded_score.append(Chord(notes=notes, duration=timestep))
    return decoded_score


map(decode.add_command, [
    #decode_chord_constant_t
])

import click
import json, cPickle
import os, glob, subprocess
import h5py

from music21 import *

from constants import *

@click.group()
def analysis():
    """Performs various analyses."""
    pass

@click.command()
def embed_note():
    text_to_utf = { v:k for k,v in json.load(open(SCRATCH_DIR + '/utf_to_txt.json', 'rb')).items() }
    utf_to_idx = json.load(open(SCRATCH_DIR + '/concat_corpus.json', 'rb'))['token_to_idx']
    in_text = []
    for utf in utf_to_idx: # we iterate across 'concat_corpus.json' because it omits unseen symbols (like the torch-rnn model)
        in_text.append(utf)
    in_text = ''.join(in_text)

    input_fp = SCRATCH_DIR + '/input.utf'
    open(input_fp, 'wb').write(in_text)
    subprocess.call(' '.join([
        'zsh',
        '~/bachbot/scripts/analysis/embed_note.zsh',
        input_fp
    ]), shell = True)

@click.command()
def embed_chords():
    text_to_utf = { v:k for k,v in json.load(open(SCRATCH_DIR + '/utf_to_txt.json', 'rb')).items() }
    utf_to_idx = json.load(open(SCRATCH_DIR + '/concat_corpus.json', 'rb'))['token_to_idx']

    chords = [
        chord.Chord(['C3','E3','G3','C4'])
    ]

    for c in chords:

        in_text = [START_DELIM]
        for note in c:
            data = (note.pitch.midi, True)
            text_data = str(data)
            in_text.append(text_to_utf[text_data])
        in_text.append(text_to_utf[CHORD_BOUNDARY_DELIM])
        in_text = ''.join(in_text)

        input_fp = SCRATCH_DIR + '/input.utf'
        open(input_fp, 'wb').write(in_text)
        subprocess.call(' '.join([
            'zsh',
            '~/bachbot/scripts/analysis/embed_note.zsh',
            input_fp
        ]), shell = True)

map(analysis.add_command, [
    embed_note,
    embed_chords
    ])

import click
import subprocess

from constants import *

@click.command()
@click.option('-i', '--infile', type=click.File('rb'), default=SCRATCH_DIR + '/concat_corpus.txt')
@click.option('-o', '--outdir', type=click.Path(exists=True), default=SCRATCH_DIR)
def make_h5(infile, outdir):
    """Encodes corpus for torch-rnn."""
    fileName=os.path.splitext(os.path.basename(infile.name))[0]

    # preprocess data (tokenize store in hdf5)
    infile_path = os.path.abspath(infile.name)
    print 'Processing corpus at: ' + infile_path
    print 'Outputting to: ' + outdir + '/' + fileName + '{h5,json}'
    subprocess.call(' '.join([
        'python',
        '~/torch-rnn/scripts/preprocess.py',
        '--input_txt', infile_path,
        '--output_h5', outdir + '/' + fileName + '.h5',
        '--output_json', outdir + '/' + fileName + '.json'
    ]), shell=True)

@click.command()
def train():
    """Trains torch-rnn model. Alias to bachbot/scripts/torchrnn/train.zsh."""
    subprocess.call(BACHBOT_DIR + '/scripts/torchrnn/train.zsh')

@click.command()
@click.argument('checkpoint', type=click.Path(exists=True), required=True)
@click.option('-t', '--temperature', type=float, default=0.9)
def sample(checkpoint, temperature):
    """Samples torch-rnn model. Calls bachbot/scripts/torchrnn/sample.zsh."""
    subprocess.call(
            BACHBOT_DIR + '/scripts/torchrnn/sample.zsh {0} {1} {2}'.format(checkpoint, temperature, START_DELIM),
            shell=True)

@click.command()
@click.argument('utf8-file', type=click.Path(exists=True))
@click.argument('json-file', type=click.File('rb'))
def postprocess_utf(utf8_file, json_file):
    """Post-process UTF encoded LSTM output of (pitch,duration) tuples back into music21."""
    import json
    from music21 import note, stream
    utf_to_txt = json.load(json_file)

    files = read_utf8(utf8_file, utf_to_txt)

    melodies = []
    for f in files:
        melody = stream.Stream()
        for note_txt in f:
            pitch, dur = note_txt.split(',')
            if pitch == u'REST':
                n = note.Rest()
            else:
                n = note.Note(pitch)
            n.duration.quarterLength = float(dur)
            melody.append(n)
        melodies.append(melody)

    for i,m in enumerate(melodies):
        out_dir = SCRATCH_DIR + '/out'
        if not os.path.exists(out_dir):
            print('Creating directory {0}'.format(out_dir))
            os.makedirs(out_dir)
        print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
        m.write('musicxml', out_dir + '/out-{0}.xml'.format(i))

def read_utf8(utf8_file, utf_to_txt):
    import codecs
    files = []
    curr_file = []
    utf8_file = codecs.open(utf8_file, "r", "utf-8")
    for symb in filter(lambda x: x != u'\n', utf8_file.read()):
        if symb == START_DELIM:
            curr_file = []
        elif symb == END_DELIM:
            files.append(curr_file)
            curr_file = []
        else:
            curr_file.append(utf_to_txt[symb])
    return files

@click.command()
@click.argument('utf8-file', type=click.Path(exists=True))
@click.argument('json-file', type=click.File('rb'))
def postprocess_utf_constant_timestep(utf8_file, json_file):
    """Post-process UTF encoded LSTM output of constant-timestep frames back into music21."""
    import json
    import codecs
    from music21 import note, stream
    utf_to_txt = json.load(json_file)

    files = []
    curr_file = []
    utf8_file = codecs.open(utf8_file, "r", "utf-8")
    for symb in filter(lambda x: x != u'\n', utf8_file.read()):
        if symb == START_DELIM:
            curr_file = []
        elif symb == END_DELIM:
            files.append(curr_file)
            curr_file = []
        else:
            curr_file.append(utf_to_txt[symb])

    melodies = []
    for f in files:
        melody = stream.Stream()
        curr_note = None
        curr_dur = 1
        for note_txt in f:
            pitch = note_txt
            if pitch[0] == NOTE_START_SYM:
                if curr_note:
                    curr_note.duration.quarterLength = float(curr_dur) / FRAMES_PER_CROTCHET
                    melody.append(curr_note)

                pitch = pitch[1:]
                if pitch == u'REST':
                    curr_note = note.Rest()
                else:
                    curr_note = note.Note(pitch)
                curr_dur = 1
            else:
                curr_dur += 1
        melodies.append(melody)

    for i,m in enumerate(melodies):
        out_dir = SCRATCH_DIR + '/out'
        if not os.path.exists(out_dir):
            print('Creating directory {0}'.format(out_dir))
            os.makedirs(out_dir)
        print('Writing {0}'.format(out_dir + '/out-{0}.xml'.format(i)))
        m.write('musicxml', out_dir + '/out-{0}.xml'.format(i))


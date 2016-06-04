import click
import codecs

from constants import *

@click.command()
@click.argument('files', nargs=-1, required=True)
@click.option('-o', '--output', type=click.File('wb'), default=SCRATCH_DIR + '/concat_corpus.txt')
def concatenate_corpus(files, output):
    """Concatenates individual files together into single corpus.

    Try `bachbot concatenate_corpus scratch/*.utf`.
    """
    print 'Writing concatenated corpus to {0}'.format(output.name)
    for fp in files:
        with open(fp, 'rb') as fd:
            output.write(START_DELIM + '\n' + fd.read().strip() + '\n' + END_DELIM + '\n')

def read_utf8(utf8_file, utf_to_txt):
    """Parses a UTF8 encoded concatenated corpus using `utf_to_txt` and returns a collection of ASCII text
    representations of notes."""
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

def write_monophonic_part(notes_txt, out_fp):
    """Writes a single part `score` List of ASCII text notes to a musicXML file."""
    melody = stream.Stream()
    for note_txt in notes_txt:
        pitch, dur = note_txt.split(',')
        if pitch == u'REST':
            n = note.Rest()
        else:
            n = note.Note(pitch)
        n.duration.quarterLength = float(dur)
        melody.append(n)

    out_dir = os.path.dirname(out_fp)
    if not os.path.exists(out_dir):
        print('Creating directory {0}'.format(out_dir))
        os.makedirs(out_dir)
    print('Writing {0}'.format(out_fp))
    melody.write('musicxml', out_fp)

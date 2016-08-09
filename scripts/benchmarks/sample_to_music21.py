import sys

sys.path.append('/home/fl350/bachbot/scripts')

from decode import to_musicxml


with open('sri_sample.txt') as fd:
    for i,line in enumerate(fd):
        chords = map(
                lambda ch: map(eval, filter(lambda c: c != '(.)', ch.split())),
                line.strip().split('|||'))
        to_musicxml(chords).write('musicxml', 'out-{}.xml'.format(i))


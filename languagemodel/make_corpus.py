import glob
import codecs

out_lines = []
for fp in glob.glob('/home/fl350/bachbot/scratch/BWV-*.txt'):
    with codecs.open(fp, 'r', 'utf-8') as fd:
        lines = map(
                lambda x: filter(lambda c: c not in ' \n', x),
                fd.readlines())
        out_lines.append(' '.join(lines[:35]))

with open('corpus.txt', 'w') as fd:
    fd.write('\n'.join(out_lines))

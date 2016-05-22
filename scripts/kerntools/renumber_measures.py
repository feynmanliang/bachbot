import click
import os.path
import music21

from constants import BACHBOT_DIR

@click.command()
@click.argument('sample-file')
@click.option('--out-file', help='File to output processed results..')
def renumber_measures(sample_file, out_file):
    """Replaces '@' measure delimiters with valid kern measure numbers."""
    if not out_file:
        out_file = BACHBOT_DIR + '/samples/' + os.path.splitext(os.path.basename(sample_file))[0] + '-bar.krn'
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
    click.echo('Writing processed results to {0}'.format(out_file))

    f = open(sample_file,"r").readlines()
    r = []
    bar = 1
    for l in f:
        if l.startswith("@"):
            if bar == 1:
                r.append("=1-\t=1-\t=1-\n")
            else:
                r.append("={bar}\t={bar}\t={bar}\n".format(bar=bar))
            bar += 1
        else:
            r.append(l)

    open(out_file, 'w+').writelines(r)

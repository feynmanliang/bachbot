#!/usr/bin/env python

import music21

def fix_measure_numbering(fpath):
    f = open(fpath,"r").readlines()
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
    return r

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Post-process LSTM output to proper kern format')
    parser.add_argument('filepath', type=str, help='the path to the file')

    args = parser.parse_args()

    # write output to `filename`-bar.krn
    outpath = args.filepath[:-4] + '-bar.krn'
    with open(outpath, 'w+') as out:
        r = fix_measure_numbering(args.filepath)
        out.writelines(r)

    # show musicXML of output
    m = music21.converter.parse(outpath)
    m.show("musicxml")

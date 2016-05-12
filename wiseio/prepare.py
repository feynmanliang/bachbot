#!/usr/bin/env python

import glob
REP="@\n"
composers = ["mozart","beethoven"]
for composer in composers:
    comp_txt = open(composer + ".txt","w")
    ll = glob.glob(dir + "/ana-music/corpus/{composer}/*.krn".format(composer=composer))
    for song in ll:
        lines = open(song,"r").readlines()
        out = []
        found_first = False
        for l in lines:
            if l.startswith("="):
                ## new measure, replace the measure with the @ sign, not part of humdrum
                out.append(REP)
                found_first = True
                continue
            if not found_first:
                ## keep going until we find the end of the header and metadata
                continue
            if l.startswith("!"):
                ## ignore comments
                continue
            out.append(l)
        comp_txt.writelines(out)
    comp_txt.close()

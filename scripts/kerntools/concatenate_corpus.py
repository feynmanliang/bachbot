import glob
import os

def concatenate_pieces(composer, comp_txt):
    """Removes metadata and concatenates all pieces by `composer` together into
    the file `comp_txt`.
    """
    ll = glob.glob("./corpus/{composer}/*.krn".format(composer=composer))
    for song in ll:
        lines = open(song,"r").readlines()
        out = strip_metadata(lines)
        comp_txt.writelines(out)

def strip_metadata(lines):
    """Strips metadata from .krn file format."""
    REP="@\n" # delimiters between measures
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
    return out

if __name__ == "__main__":
    directory = './scratch/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    composers = ["Bach+Johann"]
    for composer in composers:
        comp_txt = open(directory + "{composer}.txt".format(composer=composer),"w")
        concatenate_pieces(composer, comp_txt)
        comp_txt.close()


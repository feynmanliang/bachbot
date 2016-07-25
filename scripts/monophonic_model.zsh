#!/usr/bin/env zsh

# extract and encode monophonic scores
# NOTE: this should be done with `constants.FRAMES_PER_CROTCHET = 2` to quantize to 8th notes
# bachbot datasets prepare \
#   --keep-fermatas True \
#   --mono True \
#   --subset False

# bachbot datasets concatenate_corpus ~/bachbot/scratch/BWV-*mono*.utf -o ~/bachbot/scratch/concat_corpus_mono.txt
# bachbot make_h5 -i ~/bachbot/scratch/concat_corpus_mono.txt

# train mode
bachbot train -i ~/bachbot/scratch/concat_corpus_mono.h5 -c ~/bachbot/scratch/checkpoints/mono


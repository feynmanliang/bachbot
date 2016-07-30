#!/usr/bin/env zsh

# extract and encode monophonic scores
# NOTE: this should be done with `constants.FRAMES_PER_CROTCHET = 2` to quantize to 8th notes
bachbot datasets prepare \
  --keep-fermatas True \
  --mono True \
  --subset False

bachbot datasets concatenate_corpus ~/bachbot/scratch/BWV-*mono*.utf -o ~/bachbot/scratch/concat_corpus_mono.txt
bachbot make_h5 -i ~/bachbot/scratch/concat_corpus_mono.txt

bachbot train -i ~/bachbot/scratch/concat_corpus_mono.h5 -c ~/bachbot/scratch/checkpoints/mono
bachbot sample \
  ~/bachbot/scratch/checkpoints/mono/seq_length=16,wordvec=16,num_layers=2,rnn_size=128,dropout=0.3,batchnorm=1,lr=2e-3/checkpoint_9300.t7 \
  -t 0.8
bachbot decode sampled_stream ~/bachbot/scratch/sampled_0.8.utf


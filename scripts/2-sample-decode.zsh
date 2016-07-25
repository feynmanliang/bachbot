#!/usr/bin/env zsh

TMP=0.8

bachbot sample \
    ~/bachbot/scratch/checkpoints/seq_length=128,wordvec=32,num_layers=3,rnn_size=256,dropout=0.3,batchnorm=1,lr=2e-3/checkpoint_5250.t7 \
    -t $TMP
bachbot decode decode_utf --utf-to-txt-json ~/bachbot/scratch/utf_to_txt.json ~/bachbot/scratch/sampled_$TMP.utf

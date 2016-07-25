#!/usr/bin/env zsh

utf_file=$1
checkpoint=$2
out_path=$3

checkpoint={SCRATCH_DIR}/checkpoints/seq_length=128,wordvec=32,num_layers=3,rnn_size=256,dropout=0.3,batchnorm=1,lr=2e-3/checkpoint_5250.t7 
out_path=${SCRATCH_DIR}/harm_out/${utf_file:t}

cd ~/bachbot/scripts/harm_model/

th harmonize.lua \
  -checkpoint $checkpoint\
  -input $utf_file \
  > $fp

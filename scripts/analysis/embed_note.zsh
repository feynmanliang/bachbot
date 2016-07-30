#!/usr/bin/env zsh

cd ~/bachbot/scripts/harm_model/

mkdir -p $2

th ./embed_note.lua \
  -checkpoint ~/bachbot/scratch/checkpoints/seq_length=128,wordvec=32,num_layers=3,rnn_size=256,dropout=0.3,batchnorm=1,lr=2e-3/checkpoint_100.t7 \
  -embed_utf_file $1 \
  -out_dir $2 \
  -verbose 1 \
  -gpu 0

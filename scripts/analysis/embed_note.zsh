#!/usr/bin/zsh

cd ~/bachbot/scripts/harm_model/

mkdir -p $2

th ./embed_note.lua \
  -checkpoint ~/torch-rnn/checkpoints/seq_length=64,wordvec=64,num_layers=2,rnn_size=128,dropout=0,batchnorm=1,lr=2e-3/checkpoint_14000.t7 \
  -embed_text_file $1 \
  -out_dir $2 \
  -verbose 1 \
  -gpu 1

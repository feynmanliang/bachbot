#!/usr/bin/env zsh

cd ~/bachbot/scripts/harm_model/

mkdir -p $2

th ./embed_note.lua \
  -checkpoint ~/bachbot/best_model/checkpoint_10500.t7 \
  -embed_utf_file $1 \
  -out_dir $2 \
  -verbose 1 \
  -gpu 0

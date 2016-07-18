#!/usr/bin/env zsh

CHECKPOINT=$1
TEMP=$2
START_DELIM=$3

out_path=~/bachbot/scratch/sampled_$TEMP.utf

print temperature=$TEMP,start_text=$START_DELIM

cd ~/torch-rnn/
th sample.lua \
  -checkpoint $CHECKPOINT \
  -temperature $TEMP \
  -start_text $START_DELIM \
  -sample 1 -length 15000 \
  -gpu 0 \
  > $out_path

print "Saved to $out_path"

#!/usr/bin/env zsh

CHECKPOINT=$1
TEMP=$2
START_DELIM=$3

print temperature=$TEMP,start_text=$START_DELIM

cd ~/torch-rnn/
th sample.lua \
  -checkpoint $CHECKPOINT \
  -temperature $TEMP \
  -start_text $START_DELIM \
  -sample 1 -length 1000 \
  -gpu -1 \
  > ~/bachbot/scratch/sampled_$TEMP.utf

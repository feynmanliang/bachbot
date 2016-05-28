#!/usr/bin/env zsh

CHECKPOINT=$1
TEMP=$2
START_DELIM=$3

print temperature=$TEMP,start_text:
print $start_text

cd ~/torch-rnn/
th sample.lua \
  -checkpoint $CHECKPOINT \
  -temperature $TEMP \
  -start_text START_DELIM \
  -sample 1 -length 50000 \
  > ~/bachbot/scratch/sampled_$TEMP.txt

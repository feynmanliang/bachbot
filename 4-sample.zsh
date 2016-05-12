#!/usr/bin/env zsh

ITER=55350
TEMP=0.8

cd ./torch-rnn/
th sample.lua \
  -checkpoint cv/checkpoint_$ITER.t7 \
  -temperature $TEMP \
  -start_text "@" -sample 1 -length 15000 \
  > ../scratch/b5_$TEMP.txt

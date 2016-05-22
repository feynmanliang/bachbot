#!/usr/bin/env zsh

COMPOSER=Bach+Johann

# train model
cd ~/torch-rnn
SCRATCH_DIR=~/bachbot/scratch
th train.lua \
    -input_h5 ${SCRATCH_DIR}/${COMPOSER}.h5 \
    -input_json ${SCRATCH_DIR}/${COMPOSER}.json

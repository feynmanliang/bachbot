#!/usr/bin/zsh

COMPOSER=Bach+Johann

# train model
cd torch-rnn
th train.lua \
    -input_h5 ../scratch/${COMPOSER}.h5 \
    -input_json ../scratch/${COMPOSER}.json

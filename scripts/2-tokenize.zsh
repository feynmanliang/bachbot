#!/usr/bin/env zsh

COMPOSER=Bach+Johann

# preprocess data (tokenize store in hdf5)
python ~/torch-rnn/scripts/preprocess.py \
    --input_txt scratch/${COMPOSER}.txt \
    --output_h5 scratch/${COMPOSER}.h5 \
    --output_json scratch/${COMPOSER}.json

#!/usr/bin/zsh

COMPOSER=Bach+Johann

# preprocess data (tokenize store in hdf5)
python torch-rnn/scripts/preprocess.py \
    --input_txt scratch/${COMPOSER}.txt \
    --output_h5 scratch/${COMPOSER}.h5 \
    --output_json scratch/${COMPOSER}.json

# train model
cd torch-rnn
th train.lua \
    -input_h5 ../scratch/${COMPOSER}.h5 \
    -input_json ../scratch/${COMPOSER}.json

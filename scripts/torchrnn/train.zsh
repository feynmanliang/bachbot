#!/usr/bin/env zsh

COMPOSER=Bach+Johann

wordvec_size=64
num_layers=4
rnn_size=256

# train model
cd ~/torch-rnn
SCRATCH_DIR=~/bachbot/scratch

for wordvec_size in 16 32 64 128; do
    for num_layers in 2; do
        for rnn_size in 256; do
            fname="wordvec=${wordvec_size},num_layers=${num_layers},rnn_size=${rnn_size}"
            print $fname
            th train.lua \
                -input_h5 ${SCRATCH_DIR}/${COMPOSER}.h5 \
                -input_json ${SCRATCH_DIR}/${COMPOSER}.json \
                -wordvec_size $wordvec_size \
                -rnn_size $rnn_size \
                -num_layers $num_layers \
                -checkpoint_name $fname/checkpoint \
                -print_every 50 \
                -checkpoint_every 1000 \
                -gpu_backend cuda
        done
    done
done


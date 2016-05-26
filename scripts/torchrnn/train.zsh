#!/usr/bin/env zsh

input=concat

seq_length=500
wordvec_size=64
num_layers=2
rnn_size=256
dropout=0
batchnorm=1
lr=2e-3

# train model
cd ~/torch-rnn
SCRATCH_DIR=~/bachbot/scratch

for seq_length in 50 100 200; do
    fname="seq_length=${seq_length},wordvec=${wordvec_size},num_layers=${num_layers},rnn_size=${rnn_size},dropout=${dropout},batchnorm=${batchnorm},lr=${lr}"
    print $fname
    th train.lua \
        -input_h5 ${SCRATCH_DIR}/${input}.h5 \
        -input_json ${SCRATCH_DIR}/${input}.json \
        -seq_length $seq_length\
        -wordvec_size $wordvec_size \
        -rnn_size $rnn_size \
        -num_layers $num_layers \
        -dropout $dropout \
        -batchnorm $batchnorm \
        -learning_rate $lr \
        -checkpoint_name checkpoints/$fname/checkpoint \
        -print_every 50 \
        -checkpoint_every 1000 \
        -max_epochs 30 \
        -gpu_backend cuda \
        | tee ~/bachbot/logs/$fname.log
done

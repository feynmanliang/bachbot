#!/usr/bin/env zsh

input=concat_corpus

seq_length=16
wordvec_size=6
num_layers=1
rnn_size=512
dropout=0
batchnorm=1
lr=2e-3

# train model
cd ~/torch-rnn
SCRATCH_DIR=~/bachbot/scratch

fname="seq_length=${seq_length},\
wordvec=${wordvec_size},\
num_layers=${num_layers},\
rnn_size=${rnn_size},\
dropout=${dropout},\
batchnorm=${batchnorm},\
lr=${lr}"
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
    -max_epochs 200 \
    -gpu_backend cuda \
    | tee ~/bachbot/logs/$fname.log

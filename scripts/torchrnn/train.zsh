#!/usr/bin/env zsh

#input=concat_corpus_mono
input=$1
checkpoint_dir=$2

seq_length=128
wordvec_size=32
num_layers=3
rnn_size=256
dropout=0.3
batchnorm=1
lr=2e-3

# train model
cd ~/bachbot/scripts/harm_model

SCRATCH_DIR=~/bachbot/scratch

for seq_length in 128; do
    for wordvec_size in 32; do
        for num_layers in 3; do
            for rnn_size in 256; do
                for dropout in 0.3; do
                    fname="seq_length=${seq_length},\
wordvec=${wordvec_size},\
num_layers=${num_layers},\
rnn_size=${rnn_size},\
dropout=${dropout},\
batchnorm=${batchnorm},\
lr=${lr}"
                    print $fname
                    time th train.lua \
                        -input_h5 $input \
                        -input_json ${input:r}.json \
                        -seq_length $seq_length\
                        -wordvec_size $wordvec_size \
                        -rnn_size $rnn_size \
                        -num_layers $num_layers \
                        -dropout $dropout \
                        -batchnorm $batchnorm \
                        -learning_rate $lr \
                        -checkpoint_name $checkpoint_dir/$fname/checkpoint \
                        -batch_size 50 \
                        -print_every 50 \
                        -checkpoint_every 100 \
                        -max_epochs 100 \
                        -speed_benchmark 1\
                        -memory_benchmark 1\
                        -gpu -1 \
                        -gpu_backend cuda \
                        | tee ~/bachbot/logs/$fname.log
                done
            done
        done
    done
done

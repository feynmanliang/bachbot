#!/usr/bin/env python

import subprocess
import os

subprocess.call(["th", "train.lua",
    "-data_dir corpus/Bach+Johann.txt",
    "-rnn_size 128",
    "-num_layers 3",
    "-dropout 0.3",
    "-eval_val_every 100",
    "-checkpoint_dir cv/beethoven",
    "-gpuid -1"])

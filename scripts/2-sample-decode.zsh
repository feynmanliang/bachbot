#!/usr/bin/env zsh

TMP=0.8

bachbot sample \
    ~/bachbot/best_model/checkpoint_5200.t7 \
    -t $TMP
bachbot decode sampled_stream --utf-to-txt-json ~/bachbot/scratch/utf_to_txt.json ~/bachbot/scratch/sampled_$TMP.utf

#!/usr/bin/env zsh

bachbot datasets concatenate_corpus /home/fl350/bachbot/scratch/BWV-*nomask*fermatas.utf
bachbot make_h5
bachbot train

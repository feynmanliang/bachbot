#!/usr/bin/env zsh

bachbot datasets concatenate_corpus $(pwd)/../scratch/BWV-*nomask*fermatas.utf
bachbot make-h5
bachbot train

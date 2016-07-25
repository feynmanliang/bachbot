#!/usr/bin/env zsh
#
bachbot datasets concatenate_corpus ~/fl350/bachbot/scratch/BWV-*nomask*nofermatas*.utf
bachbot datasets make_h5
bachbot train

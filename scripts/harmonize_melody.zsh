#!/usr/bin/env zsh

fname=${1:r}
checkpoint=$2

bachbot datasets prepare_harm_input $fname.xml $fname.utf
cd ./scripts/harm_model
th harmonize.lua -checkpoint $checkpoint -input $fname.utf > ${fname}-harm.utf
bachbot decode single ${fname}-harm.utf

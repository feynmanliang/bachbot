#!/usr/bin/env zsh

concatFilePath=$1
outPath=~/bachbot/scratch
fileName=${$(basename $concatFilePath)%.*}

# preprocess data (tokenize store in hdf5)
print Processing corpus at: $concatFilePath
print Outputting to: $outPath/$fileName.\{h5,json\}
python ~/torch-rnn/scripts/preprocess.py \
     --input_txt $concatFilePath \
     --output_h5 $outPath/$fileName.h5 \
     --output_json $outPath/$fileName.json

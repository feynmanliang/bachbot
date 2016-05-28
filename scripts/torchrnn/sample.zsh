#!/usr/bin/env zsh

CHECKPOINT=$1
TEMP=$2

start_text=`cat <<EOF
<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise
  PUBLIC '-//Recordare//DTD MusicXML 2.0 Partwise//EN'
  'http://www.musicxml.org/dtds/partwise.dtd'>
<score-partwise>
  <movement-title>Music21 Fragment</movement-title>
  <identification>
    <creator type="composer">Music21</creator>
    <encoding>
      <encoding-date>2016-05-26</encoding-date>
      <software>Music21</software>
    </encoding>
  </identification>
  <defaults>
    <scaling>
      <millimeters>7</millimeters>
      <tenths>40</tenths>
    </scaling>
  </defaults>
  <part-list>
  <part id="P160d1f4b61250f17d96ae18a1f838f6d">
    <measure number="0">
      <attributes>
EOF`

print temperature=$TEMP,start_text:
print $start_text

cd ~/torch-rnn/
th sample.lua \
  -checkpoint $CHECKPOINT \
  -temperature $TEMP \
  -start_text $start_text \
  -sample 1 -length 50000 \
  > ~/bachbot/scratch/sampled_$TEMP.txt

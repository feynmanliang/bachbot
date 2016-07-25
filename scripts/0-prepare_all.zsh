#!/usr/bin/env zsh

cmd='bachbot datasets prepare --subset False'
$cmd --keep-fermatas False
$cmd --keep-fermatas True --mono True
$cmd --keep-fermatas True
for masked in '-m Soprano' '-m Alto' '-m Tenor' '-m Bass' '-m Alto -m Tenor' '-m Alto -m Tenor -m Bass'; do
  $cmd \
    --keep-fermatas True \
    $masked
done


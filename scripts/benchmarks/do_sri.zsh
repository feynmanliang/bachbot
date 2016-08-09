#!/usr/bin/env zsh

for ORDER in {1..25}; do
    ngram-count -order $ORDER -text ~/languagemodel/corpus_train.txt -lm bach_sri.lm >/dev/null 2>/dev/null
    ngram -order $ORDER -lm bach_sri.lm -ppl ~/languagemodel/corpus_train.txt -debug 2 > ~/data/srilm_train_perplexity.txt 2>/dev/null
    ngram -order $ORDER -lm bach_sri.lm -ppl ~/languagemodel/corpus_val.txt -debug 2 > ~/data/srilm_val_perplexity.txt 2>/dev/null
    print $ORDER
    tail -n1 ~/data/srilm_train_perplexity.txt
    tail -n1 ~/data/srilm_val_perplexity.txt
    #ngram -lm bach_sri.lm -gen 10 > sri_sample.txt
    #python sample_to_music21.py
done

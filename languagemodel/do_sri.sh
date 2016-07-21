ingram-count -text ~/bachbot/scratch/concat_corpus.txt -lm bach_sri.lm
ngram -lm bach_sri.lm -ppl ~/languagemodel/corpus.txt -debug 2 > ~/data/srilm_train_perplexity.txt
ngram -lm bach_sri.lm -gen 10 > sri_sample.txt
python sample_to_music21.py

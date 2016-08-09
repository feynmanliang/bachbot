#!/usr/bin/env zsh

for ORDER in {2..25}; do
    cat ~/languagemodel/corpus_train.txt |\
	./kenlm/bin/lmplz -o $ORDER > bach_kenlm.arpa 2>/dev/null
    ./kenlm/bin/query bach_kenlm.arpa < corpus_train.txt > ~/data/kenlm_train_perplexity.txt 2>/dev/null
    ./kenlm/bin/query bach_kenlm.arpa < corpus_val.txt > ~/data/kenlm_val_perplexity.txt 2>/dev/null
    print $ORDER
    tail -n3 ~/data/kenlm_train_perplexity.txt | head -n1
    tail -n3 ~/data/kenlm_val_perplexity.txt | head -n1
done

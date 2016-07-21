cat ~/languagemodel/corpus.txt |\
    ./kenlm/bin/lmplz -o 3 > bach_kenlm.arpa
./kenlm/bin/query bach_kenlm.arpa < corpus.txt > ~/data/kenlm_train_perplexity.txt

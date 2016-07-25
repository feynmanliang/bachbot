TMP=0.8;\
    bachbot sample \
    ~/torch-rnn/checkpoints/best_model/checkpoint_5500.t7 \
    -t $TMP && \
    bachbot decode decode_utf_fermata scratch/utf_to_txt.json scratch/sampled_$TMP.utf

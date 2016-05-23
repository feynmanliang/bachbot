# 5-23

* `wordvec_size=64` appears to perform best, should use for defaults in future:
	* `rnnsize=256`
	* `num_layers=2`
	* `wordvec_size=64`

# 5-22-overnight

* Training interrupted by `cudnn` recompilation
* Results suggest `val_loss` does best with `rnn_size=256`, `num_layers=2`

# 5-5

* Training on entire corpus
** BAD: kern format has K voices => each line has K space-delimited
notes
** This suggests output should be a K-dimensional vector rather than
character-by-character

* Traning on just chorales

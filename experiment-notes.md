# 5-25

* Added `extract_melody`, which extracts the 0th part from `music21.stream.Score`
  and assumes they are the melody

* Music representation:
	* Since music21 cannot output kern, use musicXML output
	* We currently include all header and dynamics info; should we strip that?

Results
=======
|seq length |wordvec size|num layers|rnn size|dropout|batchnorm|lr  |nepoch|final train loss|final val loss |
|-----------|------------|----------|--------|-------|---------|----|------|----------------|---------------|
|50         |64          |2         |256     |0      |0        |2e-3|51    |0.443295        |0.619          |
|500        |64          |2         |256     |0      |1        |2e-3|21.45 |0.4094          |0.5779         |
|500        |64          |2         |256     |0      |1        |2e-3|31.00 |0.440350        |0.572764       |
|500        |64          |2         |256     |0      |1        |1e-2|28.73 |0.287570        |0.6176         |
|50         |64          |2         |256     |0      |1        |1e-2|13.65 |0.390861        |0.6316         |




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

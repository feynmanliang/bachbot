# 5-29

* Will try:
  * Train on all voices
  * Split major/minor pieces apart
  * Model only the duration

# 5-28

* Improved preprocessing using `bachbot get_chorales`
    * Get corpus with `music21`
    * Transpose to Cmaj/Amin (is there a standard way to do this?)
    * Strip all information except `(Note+Octave|Rest, Duration)`
    * Write processed data to `bachbot/scratch/{bwv_id}-mono.txt`

## Results with new preprocessing

|seq length |wordvec size|num layers|rnn size|dropout|batchnorm|lr  |nepoch|final train loss|final val loss |
|-----------|------------|----------|--------|-------|---------|----|------|----------------|---------------|
|8          |64          |2         |256     |0      |1        |2e-3|30    |0.238247        |1.5794         |
|8          |64          |2         |128     |0      |1        |2e-3|50    |0.349           |1.367          |
|4          |64          |2         |128     |0      |1        |2e-3|50    |0.288           |1.434          |
|4          |32          |2         |128     |0      |1        |2e-3|50    |0.2527          |1.8538         |
|8          |32          |2         |32      |0      |1        |2e-3|50    |1.044           |1.191          |
|8          |32          |2         |64      |0      |1        |2e-3|50    |0.7539          |1.236          |
|8          |64          |2         |32      |0      |1        |2e-3|50    |1.027           |1.190          |
|2          |64          |2         |32      |0      |1        |2e-3|50    |0.783344        |1.25899        |
|4          |64          |2         |32      |0      |1        |2e-3|50    |1.064           |1.197          |
|8          |64          |1         |32      |0      |1        |2e-3|50    |1.022           |1.188          |
|8          |64          |1         |32      |0      |1        |2e-3|50    |1.096           |1.186          |
|8          |64          |3         |32      |0      |1        |2e-3|50    |0.989           |1.186          |
|8          |64          |3         |32      |0      |1        |2e-3|50    |0.953           |1.183          |
|8          |64          |4         |32      |0      |1        |2e-3|50    |1.0104          |1.2274         |
|8          |64          |4         |64      |0      |1        |2e-3|50    |1.0165          |1.2038         |
|8          |64          |4         |64      |0.5    |1        |2e-3|27.51 |1.392           |1.4355         |
|8          |64          |4         |64      |0.5    |0        |2e-3|25.10 |1.807           |1.851          |
|6          |64          |3         |32      |0      |1        |2e-3|50    |0.9304          |1.2137         |
|8          |64          |3         |16      |0      |1        |2e-3|50    |1.264           |1.2311         |
|12         |64          |3         |32      |0      |1        |2e-3|50    |1.030           |1.1909         |

Generative results don't sound too realistic...

## Try overfitting a model and sampling

`seq_length=8,wordvec=128,num_layers=2,rnn_size=256,dropout=0,batchnorm=1,lr=2e-3`

* Sounds much better with an overfit LSTM and `temperature=0.98`...
    * Maybe generalizable modeling isn't a good criteria... 


# 5-25

* Added `extract_melody`, which extracts the 0th part from `music21.stream.Score`
  and assumes they are the melody

* Music representation:
	* Since music21 cannot output kern, use musicXML output
	* We currently include all header and dynamics info; should we strip that?

## Results on musicXML monophonic melody

|seq length |wordvec size|num layers|rnn size|dropout|batchnorm|lr  |nepoch|final train loss|final val loss |
|-----------|------------|----------|--------|-------|---------|----|------|----------------|---------------|
|500        |64          |2         |256     |0      |1        |2e-3|16.19 | 0.022378       |0.029262       |
|50         |64          |2         |256     |0      |1        |2e-3|13.41 | 0.028490       |0.032692       |
|100        |64          |2         |256     |0      |1        |2e-3|13.41 | 0.028490       |0.032692       |



## Results on kern format data

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

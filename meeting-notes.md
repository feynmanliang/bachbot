6-24
====

Research Ideas:

* Neural Turing Machines
* Interpolating between two LSTM states

Research Tool Feedback:

* Side by side, up to 5
* Put all questions on the same page
* Music Background solicitation before page
* Amount of time spent on metric

* Buy domain name (10 quid), AWS billing limit (20 quid)

05-19 Group Meeting
===================

Visitor agreement

-	Does not apply to our work
-	For when we use MSFT, read private poster, and decide to do something off of it

[Feynman's repo](http://github.com/feynmanliang/bachbot) will be repo we collaborate on.

Weekly 1-1s

-	What did last week
-	Plan for next week
-	Bring up questions, get connected to Matt's network

Today's agenda:

-	Evaluation: how do we know what we're doing is working
-	First steps into the project

Evaluation
----------

Baseline: log probability

Subjective evaluation: completion of a given composition

-	Biases towards copying rather than creative generation

Train on untransposed, model transposed

Do we want to pursue this metric learning problem? YES

-	**AI**(Mark): will send us a list of statistics used for evaluating chorales
-	**AI**(Feynman+Marcin): Do we have metadata partitioning the piece up into certain key signature blocks?
-	Is there a corpus of Bach pastiches (Bach-like data)? Can use this for training.

Bach transcription ==> Statistics defined by Mark ==> Generative model over statistics features (probability for Bach-like)

Training data
-------------

-	Augmenting training data:
	-	Transpositions to the training set? Goal is for LSTM to not care about

Action Plan
-----------

Evaluation statistics:

1.	Mark: propose statistics
2.	Feynman+Marcin: implement in Python
3.	Construct factor graph for generative model on statistics
4.	Bring in to Microsoft, factor graph model evaluation in Infer.network

Single voice melody generation:

1.	Extract soprano lines
2.	Simple LSTM

### Longer term

Single voice melody generation

1.	Extract melody (soprano)) lines from Bach fugues
2.	Train 1-voice LSTM on
3.	Could extend to use bidirectional LSTM
4.	How to generate time/key signature? Sample before and preprocess metadata?

Chorale harmonization given the melody

1.	Generate single voice melody given first model
2.	Use the melody as an input, output the other voices
3.	Connectivity structure: current voice depends on history of current voice only and all voices at current time only

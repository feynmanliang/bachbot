# BachBot
BachBot is a research project utilizing long short term memory (LSTMs)
to generate Bach compositions.

## Installation

Docker container images are hosted on DockerHub.

CPU-only
```
docker pull fliang/bachbot:base
```

CUDA-7.5
```
docker pull fliang/bachbot:CUDA-7.5
```

## Getting Started

Set up environment
```
source scripts/activate
```

Prepare polyphonic Bach chorale corpus and train `torch-rnn` LSTM.
```
bachbot chorales prepare_poly && \
	bachbot concatenate_corpus scratch/*.utf && \
	bachbot make_h5 && \
	bachbot train
```

Sample the trained LSTM and decode output to musicXML.
```
bachbot sample <path_to_checkpoint> -t <temperature> && \
	bachbot decode decode_chord_constant_t_utf scratch/utf_to_txt.json scratch/sampled_<temperature>.utf
```

## Workflow

* `source ./scripts/activate.zsh` to set up the working environment
	* Pro-tip: Add `source ~/bachbot/scripts/activate` to your `.{bash|zsh}rc`
* To develop on the scripts
	```
	cd ./scripts
	pip install --editable .
	```
	The `bachbot` shell command will use the entry
	point defined inside `./scripts/setup.py`.
* To generate `ctags` which include system Python libraries
	```
	ctags -R -f ./tags `python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"`<CR>
	```

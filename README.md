# BachBot
BachBot is a research project utilizing long short term memory (LSTMs)
to generate Bach compositions.

## Installation

* Install submodules:

    `git submodule init`
    `git submodule update --recursive`

* Set up virtualenv:

    `source venv/bin/activate`
    `pip install -r requirements.txt`

* Set up torch:

    `cd ~/torch`
    `bash install-deps` (requires root, if not pray that you have the LuaJIT/Torch dependencies already installed)
    `./install.sh`

* Activate torch (tip: add to `.{bash,zsh}rc`):

    `./install/bin/torch-activate`

* Install torch dependencies:

    `luarocks install nngraph`
    `luarocks install optim`

## Workflow

* `./activate.zsh` sources an existing intallation

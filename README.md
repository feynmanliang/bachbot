# BachBot
BachBot is a research project utilizing long short term memory (LSTMs)
to generate Bach compositions.

## Installation

* Install submodules:

    ```
    git submodule init
    git submodule update --recursive
    ```

* Set up virtualenv:

    ```
    virtualenv -p python2.7 venv/
    source venv/bin/activate
    pip install -r requirements.txt
    ```

* Set up torch:

    ```
    cd ~/torch
    bash install-deps
    ./install.sh
    ```

* Activate torch (tip: add to `.{bash,zsh}rc`):

    `./install/bin/torch-activate`

* Install `torch-rnn` dependencies:

    ```
    sudo apt-get -y install python2.7-dev
    sudo apt-get install libhdf5-dev

    # Install most things using luarocks
    luarocks install torch
    luarocks install nn
    luarocks install optim
    luarocks install lua-cjson

    # We need to install torch-hdf5 from GitHub
    git clone https://github.com/deepmind/torch-hdf5
    cd torch-hdf5
    luarocks make hdf5-0-0.rockspec
    ```
** I had to use `HDF5_ROOT=/home/fl350/usr luarocks make
hdf5-0-0.rockspec` because I local-installed `hdf5` to `~/usr`

* For GPU acceleration with CUDA, you'll need CUDA > 6.5 and:

    ```
    luarocks install cutorch
    luarocks install cunn
    ```

## Workflow

* `source ./activate.zsh` sources an existing intallation

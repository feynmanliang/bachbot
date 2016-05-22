############################################################
# Dockerfile for the BachBot project
# Based on Ubuntu
#
# Building, pushing, and running:
#   docker build -f Dockerfile -t bachbot:CUDA-7.5 .
#   docker tag -f <tag of last container> fliang/bachbot:CUDA-7.5
#   docker push fliang/bachbot:CUDA-7.5
#   docker run -i -t fliang/bachbot:CUDA-7.5
############################################################

FROM nvidia/cuda:7.5
MAINTAINER Feynman Liang "feynman.liang@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Required packages
RUN apt-get update
RUN apt-get -y install \
    python \
    build-essential \
    python2.7-dev \
    git \
    ssh \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    software-properties-common

#pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | /bin/bash
RUN pyenv install 2.7.11

# Torch and luarocks
RUN git clone https://github.com/torch/distro.git ~/torch --recursive && cd ~/torch && \
    bash install-deps && \
    ./install.sh -b

ENV LUA_PATH='~/.luarocks/share/lua/5.1/?.lua;~/.luarocks/share/lua/5.1/?/init.lua;~/torch/install/share/lua/5.1/?.lua;~/torch/install/share/lua/5.1/?/init.lua;./?.lua;~/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='~/.luarocks/lib/lua/5.1/?.so;~/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=~/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=~/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=~/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='~/torch/install/lib/?.so;'$LUA_CPATH

#BachBot
WORKDIR ~/bachbot
RUN git clone https://github.com/feynmanliang/bachbot.git ~/bachbot

#BachBot virtualenv
WORKDIR ~/bachbot
RUN pyenv virtualenv 2.7.11 bachbot && \
    /bin/bash -c "pyenv activate bachbot && \
    pip install -r requirements.txt"

#Lua requirements
WORKDIR ~
RUN luarocks install torch
RUN luarocks install nn
RUN luarocks install optim
RUN luarocks install lua-cjson

RUN git clone https://github.com/deepmind/torch-hdf5 ~/torch-hdf5
WORKDIR ~/torch-hdf5
RUN luarocks make hdf5-0-0.rockspec

#torch-rnn and python requirements installed to bachbot venv
WORKDIR ~
RUN git clone https://github.com/jcjohnson/torch-rnn && \
    /bin/bash -c "pyenv activate bachbot && \
    pip install -r torch-rnn/requirements.txt"

#Element-Research/rnn
WORKDIR ~
RUN luarocks install rnn

#CUDA
WORKDIR ~
RUN luarocks install cutorch
RUN luarocks install cunn
RUN luarocks install cunnx

# Clean tmps
RUN apt-get clean && \
    rm -rf \
	/var/lib/apt/lists/* \
	/tmp/* \
	/var/tmp/* \
	~/torch-hdf5

##################### INSTALLATION END #####################
WORKDIR ~/
ENTRYPOINT bash

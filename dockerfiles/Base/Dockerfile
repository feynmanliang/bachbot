############################################################
# Dockerfile for the BachBot project
# Based on Ubuntu
#
# Building, pushing, and running:
#   docker build -f Dockerfile -t bachbot:base .
#   docker tag <tag of last container> fliang/bachbot:base
#   docker push fliang/bachbot:base
#   docker run -it --net=host fliang/bachbot:base
# visit localhost:8888
############################################################

FROM ubuntu:14.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
MAINTAINER Feynman Liang "feynman.liang@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# Required packages
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    python2.7-dev \
    git \
    ssh \
    libhdf5-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    software-properties-common \
    vim \
    pkg-config \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    python-numpy \
    python-scipy \
    python-matplotlib \
    python-nose \
    libbz2-dev \
    libfreetype6-dev \
    zsh

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

#Lua requirements
RUN luarocks install nn
RUN luarocks install optim
RUN luarocks install lua-cjson

# Torch-hdf5
RUN git clone https://github.com/deepmind/torch-hdf5 ~/torch-hdf5 && \
    cd ~/torch-hdf5 && \
    luarocks make hdf5-0-0.rockspec

# pip
RUN apt-get -y install python-pip
RUN pip install --upgrade pip

#torch-rnn and python requirements
RUN luarocks install luautf8
RUN git clone https://github.com/feynmanliang/torch-rnn ~/torch-rnn
RUN pip install -r ~/torch-rnn/requirements.txt

#Element-Research/rnn
# RUN luarocks install rnn

#BachBot
RUN git clone https://github.com/feynmanliang/bachbot.git ~/bachbot
RUN cd ~/bachbot && \
    pip install -r requirements.txt && \
    pip install --editable scripts

# Clean tmps
RUN apt-get clean && \
    rm -rf \
	/var/lib/apt/lists/* \
	/tmp/* \
	/var/tmp/* \
	~/torch-hdf5

# Make Required DIRs
RUN mkdir ~/bachbot/scratch
RUN mkdir ~/bachbot/scratch/out
RUN mkdir ~/bachbot/logs
##################### INSTALLATION END #####################
EXPOSE 8888
WORKDIR /root/bachbot
COPY start.sh .
#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
CMD ["./start.sh"]

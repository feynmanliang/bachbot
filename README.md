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

## Workflow

* `source ./scripts/activate.zsh` to set up the working environment

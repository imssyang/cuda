#!/bin/bash

export CUDA_DISABLE_ENV=yes
export CUDA_ROOT=/usr/local/cuda/bin
eval "optbin -s /opt/cuda/bin"
eval "optlib -s /opt/cuda/lib64"


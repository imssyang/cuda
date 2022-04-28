#!/bin/bash

export CUDA_DISABLE_ENV=yes
export CUDA_ROOT=/opt/cuda
export CPATH=$CPATH:/opt/cuda/include
export LIBRARY_PATH=$LIBRARY_PATH:/opt/cuda/lib64
eval "optbin -s /opt/cuda/bin"
eval "optlib -s /opt/cuda/lib64"


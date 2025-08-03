#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
INSTALL_DIR=/home/jeromeku/cupti_profiler/kineto/libkineto/install
CUDA_HOME=/home/jeromeku/cuda-toolkit
ls -lth ${CUDA_HOME}/extras/CUPTI/lib64 | grep cupti


g++ \
  -g3 \
  -O0 \
  kineto_playground.cpp \
  -o main \
  -I/home/jeromeku/cuda-toolkit/include \
  -I../third_party/fmt/include \
  -I${INSTALL_DIR}/include/kineto \
  -L/usr/lib \
  -L${CUDA_HOME}/lib64 \
  -L${CUDA_HOME}/extras/CUPTI/lib64 \
  -lpthread \
  -lcuda \
  -lcudart \
  -lcupti \
  -lnvperf_host \
  ${INSTALL_DIR}/lib/libkineto.a \
  kplay_cu.o

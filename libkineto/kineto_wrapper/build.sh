#!/bin/bash
rm -r build
KINETO_HOME=/home/jeromeku/cupti_profiler/kineto/libkineto/install
# CUDA_HOME=/home/jeromeku/cuda-toolkit
PYBIND_HOME=/home/jeromeku/cupti_profiler/.cupti-env/lib/python3.12/site-packages/pybind11/share/cmake
FLAGS=("-DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
FLAGS+=("-DKINETO_PREFIX=${KINETO_HOME}")
# FLAGS+=("-DCUDA_HOME=${CUDA_HOME}")
FLAGS+=("-DCMAKE_PREFIX_PATH=${PYBIND_HOME}")
DEBUG_INFO="-g1"
FLAGS+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=${DEBUG_INFO}")
# FLAGS+=("-DCUDAToolkit_ROOT=/home/jeromeku/cuda-toolkit")

CMD="cmake -Bbuild -S. -GNinja ${FLAGS[@]}"
echo "${CMD}"
eval "${CMD}"
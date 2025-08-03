#!/bin/bash

set -euo pipefail

rm -rf build

CMD="cmake -GNinja -Bbuild -S. \
-DCMAKE_VERBOSE_MAKEFILE=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCUDA_SOURCE_DIR=/home/jeromeku/cuda-toolkit \
-DKINETO_BUILD_TESTS=OFF \
-DLIBKINETO_NOROCTRACER=ON \
-DCMAKE_CXX_STANDARD=20 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON \
-DKINETO_LIBRARY_TYPE=shared"

echo ">> ${CMD}"
eval "${CMD}"
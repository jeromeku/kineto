#!/bin/bash

CMD="cmake -Bbuild -S. -DCMAKE_VERBOSE_MAKEFILE=ON \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-GNinja \
--trace-expand"

echo "${CMD}"
eval "${CMD}"
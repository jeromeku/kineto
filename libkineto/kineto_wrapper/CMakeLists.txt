cmake_minimum_required(VERSION 3.22)
project(pykineto LANGUAGES CXX)
message(STATUS "CUDA HOME ${CUDA_HOME} KINETO_PREFIX ${KINETO_PREFIX}")

# Where libkineto, pybind11 and CUDA live
list(PREPEND CMAKE_PREFIX_PATH "${KINETO_PREFIX}/")
# Tell CMake where the toolkit really lives *before* find_package().
# (replace with the path that contains bin/, lib64/, extras/CUPTI/, …)
# set(CUDAToolkit_ROOT "/home/jeromeku/cuda-toolkit")

find_package(CUDAToolkit REQUIRED COMPONENTS cupti)  # now picks the toolkit copy
find_package(kineto       REQUIRED CONFIG NAMES kinetoLibrary kineto)
find_package(pybind11     REQUIRED CONFIG)

get_target_property(_cupti_path CUDA::cupti LOCATION)
message(STATUS "CUDA::cupti = ${_cupti_path}")

pybind11_add_module(pykineto wrapper.cpp)

target_link_libraries(pykineto PRIVATE
    kineto
    CUDA::cudart CUDA::cuda_driver)

# Runtime search path: directory of CUDA::cupti  +  $ORIGIN (for libkineto.so)
# set_target_properties(pykineto PROPERTIES
#     BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cupti>"
#     INSTALL_RPATH "$ORIGIN")
    
target_include_directories(pykineto PUBLIC 
    "${KINETO_PREFIX}/include"
    "/home/jeromeku/cupti_profiler/kineto/libkineto/third_party/fmt/include")

# Libkineto is C++20; propagate the flag to the extension build
set_target_properties(pykineto PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES)

# RPATH so the module finds libcupti.so without LD_LIBRARY_PATH
# set_target_properties(pykineto PROPERTIES
#     BUILD_RPATH "${CUDA_HOME}/lib64;${CUDA_HOME}/extras/CUPTI/lib64")
target_link_libraries(pykineto PRIVATE
    kineto            # now this is the shared lib
    CUDA::cudart CUDA::cuda_driver)

# After the extension is built, copy the shared library next to it
add_custom_command(TARGET pykineto POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:kineto>
            $<TARGET_FILE_DIR:pykineto>)

# sample/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(kineto_playground LANGUAGES CXX)

# Point CMake at your manual install
list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/cupti_profiler/kineto/libkineto/install")

find_package(kineto REQUIRED CONFIG NAMES kinetoLibrary kineto)
find_package(CUDAToolkit REQUIRED)   # brings in CUDA::cupti etc.
get_property(all_imported GLOBAL PROPERTY IMPORTED_TARGETS)
list(SORT all_imported)
message(STATUS "Imported targets:\n  ${all_imported}")

add_executable(playground kineto_playground.cpp kplay_cu.o)
target_include_directories(playground PUBLIC
    "$ENV{HOME}/cupti_profiler/kineto/libkineto/install/include/kineto"
    "/home/jeromeku/cupti_profiler/kineto/libkineto/third_party/fmt/include")

target_link_libraries(playground PRIVATE
    kineto             # brings static libkineto.a
    CUDA::cupti CUDA::cudart CUDA::cuda_driver
)
set_target_properties(playground PROPERTIES CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.22)
project(dump_targets LANGUAGES NONE)

list(PREPEND CMAKE_PREFIX_PATH
     "$ENV{HOME}/cupti_profiler/kineto/libkineto/install/share/cmake/kineto")

find_package(kinetoLibrary CONFIG REQUIRED)   #  ➜ imports target  kineto
find_package(CUDAToolkit  REQUIRED)           #  ➜ imports targets CUDA::cudart …

# directory-local imported targets
get_directory_property(dir_imps IMPORTED_TARGETS)
list(SORT dir_imps)
message(STATUS "Directory-scope imported targets:\n  ${dir_imps}")

# (optional) global imported targets
get_property(global_imps GLOBAL PROPERTY IMPORTED_GLOBAL_TARGETS)
if(global_imps)
  list(SORT global_imps)
  message(STATUS "Global-scope imported targets:\n  ${global_imps}")
endif()

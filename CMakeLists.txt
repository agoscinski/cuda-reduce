cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Custom settings for alex workstation to link libtorch and cuda
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

project(example_reduce CXX)
include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(reduce_cpp SHARED
    src/reduce.cpp
    src/bindings.cpp
    src/reduce.hh
    src/reduce_autograd.cpp
    src/reduce_cpu.cpp
    src/reduce_cuda.cu
)
target_link_libraries(reduce_cpp PUBLIC torch)

add_executable(example_reduce example_reduce.cpp)
target_link_libraries(example_reduce reduce_cpp)

add_executable(use-torch-model use-torch-model.cpp)
target_link_libraries(use-torch-model reduce_cpp)

# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example_reduce
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example_reduce>)
# endif (MSVC)

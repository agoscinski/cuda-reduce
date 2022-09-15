cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Custom settings for alex workstation to link libtorch and cuda
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

project(example_reduce CXX CUDA)
enable_language(CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example_reduce example_reduce.cpp)
target_link_libraries(example_reduce "${TORCH_LIBRARIES}")
set_property(TARGET example_reduce PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example_reduce
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example_reduce>)
endif (MSVC)

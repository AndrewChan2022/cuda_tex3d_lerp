cmake_minimum_required(VERSION 3.18)

# project
project(lerpdemo)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(lerpdemo tex3dlerp.cu)

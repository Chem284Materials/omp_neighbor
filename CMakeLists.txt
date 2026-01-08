cmake_minimum_required(VERSION 3.22)

# Project name and version
project(NeighborMD VERSION 1.0
        DESCRIPTION "Simple example of using OpenMP"
        LANGUAGES CXX)

# Add the executable
add_executable(neighbormd src/main.cpp)

# Specify the C++ standard
target_compile_features(neighbormd PUBLIC cxx_std_11)

# Find OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(neighbormd PUBLIC OpenMP::OpenMP_CXX)


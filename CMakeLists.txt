cmake_minimum_required(VERSION 3.20)
project(distributed_parameter_server LANGUAGES CXX)

# Interface-only target for the architecture stage. No server is built yet.
add_library(eps_contracts INTERFACE)
target_include_directories(eps_contracts INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(eps_contracts INTERFACE cxx_std_20)

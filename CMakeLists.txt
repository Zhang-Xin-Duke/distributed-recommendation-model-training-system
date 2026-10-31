cmake_minimum_required(VERSION 3.20)
project(distributed_parameter_server LANGUAGES CXX)

add_library(eps_contracts INTERFACE)
target_include_directories(eps_contracts INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(eps_contracts INTERFACE cxx_std_20)

find_package(Threads REQUIRED)
add_library(eps_core
  src/store/shard_store.cpp
  src/store/snapshot.cpp
  src/access/batch_access.cpp
  src/update/lazy_adam.cpp
  src/client/distributed_client.cpp
  src/cache/access_cost_model.cpp
  src/server/service.cpp)
target_link_libraries(eps_core PUBLIC eps_contracts Threads::Threads)
target_include_directories(eps_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(eps_core PRIVATE -Wall -Wextra -Wpedantic)

option(EPS_BUILD_TESTS "Build CPU correctness checks" ON)
option(EPS_BUILD_BRPC "Build the BRPC server and client" ON)
option(EPS_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
if(EPS_SANITIZERS)
  target_compile_options(eps_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(eps_core PUBLIC -fsanitize=address,undefined)
endif()

if(EPS_BUILD_BRPC)
  if(EPS_SANITIZERS)
    message(FATAL_ERROR "Use a separate EPS_BUILD_BRPC=OFF sanitizer build for the core; the installed BRPC/Protobuf binaries are not sanitizer-built.")
  endif()
  find_package(Protobuf CONFIG QUIET)
  if(NOT TARGET protobuf::libprotobuf)
    find_package(Protobuf REQUIRED)
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(BRPC REQUIRED IMPORTED_TARGET brpc)
  find_package(nlohmann_json CONFIG REQUIRED)
  add_library(eps_proto proto/embedding_service.proto)
  target_link_libraries(eps_proto PUBLIC protobuf::libprotobuf)
  protobuf_generate(TARGET eps_proto APPEND_PATH)
  target_include_directories(eps_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
  add_library(eps_brpc src/transport/brpc_client.cpp src/transport/brpc_server.cpp)
  target_link_libraries(eps_brpc PUBLIC eps_core eps_proto PkgConfig::BRPC)
  target_include_directories(eps_brpc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_executable(eps_server src/server/main.cpp)
  target_link_libraries(eps_server PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  target_include_directories(eps_server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  if(EPS_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(eps_rpc_checks tests/rpc_checks.cpp)
    target_link_libraries(eps_rpc_checks PRIVATE eps_brpc)
    target_include_directories(eps_rpc_checks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME two_process_rpc_and_restore
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/rpc_processes.py"
              $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_rpc_checks>)
    set_tests_properties(two_process_rpc_and_restore PROPERTIES TIMEOUT 90)
  endif()
endif()
if(EPS_BUILD_TESTS)
  enable_testing()
  add_executable(eps_core_tests tests/core_tests.cpp)
  target_link_libraries(eps_core_tests PRIVATE eps_core)
  add_test(NAME core_correctness COMMAND eps_core_tests)
  add_executable(eps_planner_tests tests/planner_tests.cpp)
  target_link_libraries(eps_planner_tests PRIVATE eps_core)
  add_test(NAME cost_and_migration_policies COMMAND eps_planner_tests)
endif()

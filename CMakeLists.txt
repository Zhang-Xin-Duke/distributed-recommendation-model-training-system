cmake_minimum_required(VERSION 3.20)
project(distributed_parameter_server LANGUAGES CXX)
option(EPS_BUILD_RUNTIME "Build the versioned native C runtime" OFF)
if(EPS_BUILD_RUNTIME)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_library(eps_contracts INTERFACE)
target_include_directories(eps_contracts INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(eps_contracts INTERFACE cxx_std_20)

find_package(Threads REQUIRED)
add_library(eps_core
  src/store/shard_store.cpp
  src/store/snapshot.cpp
  src/access/batch_access.cpp
  src/access/prepared_lookup.cpp
  src/access/prepare_queue.cpp
  src/access/lookup_layout_cache.cpp
  src/access/gradient_layout_cache.cpp
  src/access/compact_gradients.cpp
  src/update/lazy_adam.cpp
  src/training/training_progress.cpp
  src/training/synchronous_gradients.cpp
  src/replication/record.cpp
  src/replication/log.cpp
  src/client/distributed_client.cpp
  src/client/cache_resolver.cpp
  src/cache/access_cost_model.cpp
  src/cache/reuse_policy.cpp
  src/cache/host_backend.cpp
  src/cache/agent_runtime.cpp
  adapters/entropy/table_adapter.cpp
  src/server/service.cpp)
target_link_libraries(eps_core PUBLIC eps_contracts Threads::Threads)
target_include_directories(eps_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(eps_core PRIVATE -Wall -Wextra -Wpedantic)

option(EPS_BUILD_TESTS "Build CPU correctness checks" ON)
option(EPS_BUILD_BRPC "Build the BRPC server and client" ON)
option(EPS_BUILD_CUDA "Build the CUDA cache backend" OFF)
option(EPS_BUILD_PYTHON "Build the Python/TensorFlow embedding adapter" OFF)
if(EPS_BUILD_RUNTIME AND NOT EPS_BUILD_BRPC)
  message(FATAL_ERROR "The native C runtime requires EPS_BUILD_BRPC=ON.")
endif()
option(EPS_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
if(EPS_SANITIZERS)
  target_compile_options(eps_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(eps_core PUBLIC -fsanitize=address,undefined)
endif()

if(EPS_BUILD_PYTHON)
  if(NOT EPS_BUILD_BRPC)
    message(FATAL_ERROR "The Python adapter requires EPS_BUILD_BRPC=ON.")
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set_target_properties(eps_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE eps_pybind_directory OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  list(APPEND CMAKE_PREFIX_PATH "${eps_pybind_directory}")
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(EPS_BUILD_CUDA)
  if(EPS_SANITIZERS)
    message(FATAL_ERROR "Use a separate core-only sanitizer build; CUDA checks run in a native CUDA build.")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(eps_cuda src/cuda/cache_backend.cu)
  target_link_libraries(eps_cuda PUBLIC eps_core CUDA::cudart)
  target_include_directories(eps_cuda PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set_target_properties(eps_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(eps_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

if(EPS_BUILD_BRPC)
  if(EPS_SANITIZERS)
    message(FATAL_ERROR "Use a separate EPS_BUILD_BRPC=OFF sanitizer build for the core; the installed BRPC/Protobuf binaries are not sanitizer-built.")
  endif()
  if(EPS_BUILD_RUNTIME AND UNIX AND NOT APPLE)
    set(Protobuf_USE_STATIC_LIBS ON)
  else()
    find_package(Protobuf CONFIG QUIET)
  endif()
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
  add_library(eps_brpc src/transport/brpc_client.cpp src/transport/brpc_server.cpp
    src/transport/brpc_replica_client.cpp src/transport/brpc_replica_server.cpp
    src/transport/replicated_rounds.cpp
    src/transport/brpc_training_client.cpp src/transport/brpc_training_server.cpp
    src/transport/brpc_synchronous_client.cpp src/transport/brpc_synchronous_server.cpp
    src/transport/brpc_cache_client.cpp src/transport/brpc_agent_server.cpp
    src/transport/worker_cache_agent.cpp)
  if(EPS_BUILD_RUNTIME AND UNIX AND NOT APPLE)
    find_library(EPS_BRPC_STATIC NAMES libbrpc.a HINTS ${BRPC_LIBRARY_DIRS} REQUIRED)
    find_library(EPS_PROTOC_STATIC NAMES libprotoc.a REQUIRED)
    set(eps_private_brpc_deps ${BRPC_STATIC_LIBRARIES})
    list(REMOVE_ITEM eps_private_brpc_deps brpc protobuf protoc z)
    find_package(ZLIB REQUIRED)
    target_include_directories(eps_brpc PUBLIC ${BRPC_INCLUDE_DIRS})
    target_link_directories(eps_brpc PUBLIC ${BRPC_STATIC_LIBRARY_DIRS})
    # BRPC's util/type-resolver references need the protobuf archive after BRPC
    # and protoc, rather than only through the earlier eps_proto dependency.
    target_link_libraries(eps_brpc PUBLIC eps_core eps_proto "${EPS_BRPC_STATIC}" "${EPS_PROTOC_STATIC}" ${eps_private_brpc_deps} protobuf::libprotobuf ZLIB::ZLIB)
  else()
    target_link_libraries(eps_brpc PUBLIC eps_core eps_proto PkgConfig::BRPC)
  endif()
  # A static BRPC installation exposes its transitive dependencies through
  # Libs.private. The ordinary pkg-config imported target omits those entries.
  foreach(eps_brpc_library IN LISTS BRPC_LINK_LIBRARIES)
    if(EPS_BUILD_RUNTIME AND UNIX AND NOT APPLE)
      break()
    endif()
    if(eps_brpc_library MATCHES "\\.a$")
      target_link_directories(eps_brpc PUBLIC ${BRPC_STATIC_LIBRARY_DIRS})
      target_link_libraries(eps_brpc PUBLIC ${BRPC_STATIC_LIBRARIES})
      break()
    endif()
  endforeach()
  target_include_directories(eps_brpc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_library(eps_redis_compat adapters/entropy/redis-sdk/src/client.cpp)
  set_target_properties(eps_redis_compat PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_compile_features(eps_redis_compat PUBLIC cxx_std_17)
  target_include_directories(eps_redis_compat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy/redis-sdk/include")
  target_link_libraries(eps_redis_compat PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
  if(EPS_BUILD_RUNTIME AND UNIX AND NOT APPLE)
    target_include_directories(eps_redis_compat PUBLIC ${BRPC_INCLUDE_DIRS})
    target_link_directories(eps_redis_compat PUBLIC ${BRPC_STATIC_LIBRARY_DIRS})
    target_link_libraries(eps_redis_compat PUBLIC "${EPS_BRPC_STATIC}" "${EPS_PROTOC_STATIC}" ${eps_private_brpc_deps} protobuf::libprotobuf ZLIB::ZLIB)
  else()
    target_link_libraries(eps_redis_compat PUBLIC PkgConfig::BRPC protobuf::libprotobuf)
    foreach(eps_redis_library IN LISTS BRPC_LINK_LIBRARIES)
      if(eps_redis_library MATCHES "\\.a$")
        target_link_directories(eps_redis_compat PUBLIC ${BRPC_STATIC_LIBRARY_DIRS})
        target_link_libraries(eps_redis_compat PUBLIC ${BRPC_STATIC_LIBRARIES})
        break()
      endif()
    endforeach()
  endif()
  if(EPS_BUILD_RUNTIME)
    add_library(eps_runtime SHARED adapters/entropy/runtime_c.cpp)
    target_link_libraries(eps_runtime PRIVATE eps_brpc nlohmann_json::nlohmann_json)
    target_include_directories(eps_runtime PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(eps_runtime PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    if(APPLE)
      target_link_options(eps_runtime PRIVATE "-Wl,-exported_symbol,_eps_runtime_api_v1")
    elseif(UNIX)
      target_link_options(eps_runtime PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy/runtime.exports" "-Wl,-Bsymbolic" "-Wl,--exclude-libs,ALL" "-Wl,-z,defs")
    endif()
    if(EPS_BUILD_CUDA)
      target_compile_definitions(eps_runtime PRIVATE EPS_HAS_CUDA=1)
      target_link_libraries(eps_runtime PRIVATE eps_cuda)
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import importlib.util; assert importlib.util.find_spec('tensorflow')"
      RESULT_VARIABLE eps_native_tf_available OUTPUT_QUIET ERROR_QUIET)
    if(eps_native_tf_available EQUAL 0)
      set(eps_tf_operations "${CMAKE_CURRENT_BINARY_DIR}/eps_tensorflow_ops.so")
      add_custom_command(OUTPUT "${eps_tf_operations}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_tensorflow_ops.py" --out "${eps_tf_operations}"
        DEPENDS adapters/entropy/tensorflow_ops.cc include/eps/runtime_c.h scripts/build_tensorflow_ops.py VERBATIM)
      add_custom_target(eps_tensorflow_operations ALL DEPENDS "${eps_tf_operations}")
      if(EPS_BUILD_TESTS)
        if(APPLE)
          add_test(NAME native_tensorflow_platform_guard COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_tensorflow_platform.py" "${eps_tf_operations}" $<TARGET_FILE:eps_runtime>)
        else()
          add_test(NAME native_tensorflow_cpu_graph COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_tensorflow_processes.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime> "${eps_tf_operations}" $<TARGET_FILE:eps_snapshot_compare> cpu)
          set_tests_properties(native_tensorflow_cpu_graph PROPERTIES TIMEOUT 180 ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          add_test(NAME native_session_context_cpu COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_session_context.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime> "${eps_tf_operations}" cpu)
          set_tests_properties(native_session_context_cpu PROPERTIES TIMEOUT 180 ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          add_test(NAME original_runner_native_training_progress_cpu COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_training_progress.py" --build "${CMAKE_CURRENT_BINARY_DIR}"
            --runtime $<TARGET_FILE:eps_runtime> --operations "${eps_tf_operations}"
            --out "${CMAKE_CURRENT_BINARY_DIR}/original-runner-training-progress-cpu.json")
          set_tests_properties(original_runner_native_training_progress_cpu PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          if(EPS_BUILD_CUDA)
            add_test(NAME native_tensorflow_cuda_graph COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_tensorflow_processes.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime> "${eps_tf_operations}" $<TARGET_FILE:eps_snapshot_compare> cuda)
            set_tests_properties(native_tensorflow_cuda_graph PROPERTIES TIMEOUT 180 ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
            add_test(NAME native_session_context_cuda COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_session_context.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime> "${eps_tf_operations}" cuda)
            set_tests_properties(native_session_context_cuda PROPERTIES TIMEOUT 180 ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
            add_test(NAME original_runner_native_training_progress_cuda COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_training_progress.py" --build "${CMAKE_CURRENT_BINARY_DIR}"
              --runtime $<TARGET_FILE:eps_runtime> --operations "${eps_tf_operations}" --model-device cuda
              --out "${CMAKE_CURRENT_BINARY_DIR}/original-runner-training-progress-cuda.json")
            set_tests_properties(original_runner_native_training_progress_cuda PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77
              ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          endif()
        endif()
      endif()
    endif()
    if(EPS_BUILD_TESTS)
      add_test(NAME native_runtime_c_abi COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_c_processes.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime>)
      set_tests_properties(native_runtime_c_abi PROPERTIES TIMEOUT 45)
      add_test(NAME native_runtime_training_progress COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_training_progress.py" --build "${CMAKE_CURRENT_BINARY_DIR}"
        --runtime $<TARGET_FILE:eps_runtime> --out "${CMAKE_CURRENT_BINARY_DIR}/native-training-progress.json")
      set_tests_properties(native_runtime_training_progress PROPERTIES TIMEOUT 120)
      add_test(NAME native_runtime_owner_incarnation COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_c_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_runtime> $<TARGET_FILE:eps_cache_agent>)
      set_tests_properties(native_runtime_owner_incarnation PROPERTIES TIMEOUT 45)
    endif()
  endif()
  if(EPS_BUILD_PYTHON)
    set_target_properties(eps_proto eps_brpc PROPERTIES POSITION_INDEPENDENT_CODE ON)
    pybind11_add_module(_eps_native adapters/entropy/python_bindings.cpp)
    target_link_libraries(_eps_native PRIVATE eps_brpc nlohmann_json::nlohmann_json)
    target_include_directories(_eps_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(_eps_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
  endif()
  add_executable(eps_server src/server/main.cpp)
  target_link_libraries(eps_server PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  target_include_directories(eps_server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_executable(eps_training_coordinator src/server/training_main.cpp)
  target_link_libraries(eps_training_coordinator PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  target_include_directories(eps_training_coordinator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_executable(eps_training_worker examples/sparse_training_worker.cpp)
  target_link_libraries(eps_training_worker PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  find_package(OpenSSL REQUIRED)
  add_executable(eps_training_control src/training/control_main.cpp src/training/job_checkpoint.cpp)
  add_executable(eps_replica src/server/replica_main.cpp)
  target_link_libraries(eps_replica PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  target_include_directories(eps_replica PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_executable(eps_replica_control src/replication/control_main.cpp)
  target_link_libraries(eps_replica_control PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  target_include_directories(eps_replica_control PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(eps_training_control PRIVATE eps_brpc nlohmann_json::nlohmann_json OpenSSL::Crypto)
  if(EPS_BUILD_TESTS)
    add_executable(eps_read_fault_proxy tests/read_fault_proxy.cpp)
    target_link_libraries(eps_read_fault_proxy PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  endif()
  add_executable(eps_training_prepare src/training/prepare_main.cpp)
  target_link_libraries(eps_training_prepare PRIVATE eps_brpc nlohmann_json::nlohmann_json OpenSSL::Crypto)
  target_include_directories(eps_training_prepare PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_executable(eps_capacity_probe tests/capacity_probe.cpp)
  target_link_libraries(eps_capacity_probe PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  add_executable(eps_cache_policy_probe tests/cache_policy_probe.cpp)
  target_link_libraries(eps_cache_policy_probe PRIVATE eps_brpc nlohmann_json::nlohmann_json)
  add_executable(eps_cache_agent src/server/agent_main.cpp)
  target_link_libraries(eps_cache_agent PRIVATE eps_brpc)
  target_include_directories(eps_cache_agent PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  if(EPS_BUILD_CUDA)
    target_compile_definitions(eps_cache_agent PRIVATE EPS_HAS_CUDA=1)
    target_link_libraries(eps_cache_agent PRIVATE eps_cuda)
  endif()
  if(EPS_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(eps_redis_compat_tests tests/redis_compat_tests.cpp)
    target_link_libraries(eps_redis_compat_tests PRIVATE eps_redis_compat nlohmann_json::nlohmann_json)
    find_program(EPS_REDIS_SERVER redis-server)
    if(EPS_REDIS_SERVER)
      add_test(NAME public_redis_sdk COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/redis_compat_processes.py"
        --client $<TARGET_FILE:eps_redis_compat_tests> --redis-server "${EPS_REDIS_SERVER}")
      set_tests_properties(public_redis_sdk PROPERTIES TIMEOUT 45)
    else()
      message(STATUS "Redis process check requires EPS_REDIS_SERVER pointing to redis-server")
    endif()
    add_executable(eps_training_rpc_tests tests/training_rpc_tests.cpp)
    target_link_libraries(eps_training_rpc_tests PRIVATE eps_brpc)
    target_include_directories(eps_training_rpc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME training_progress_rpc COMMAND eps_training_rpc_tests)
    set_tests_properties(training_progress_rpc PROPERTIES TIMEOUT 30)
    add_test(NAME distributed_training_processes
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_training_coordinator>
        $<TARGET_FILE:eps_training_worker> $<TARGET_FILE:eps_training_control>)
    set_tests_properties(distributed_training_processes PROPERTIES TIMEOUT 180)
    add_test(NAME synchronous_training_processes
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/synchronous_training_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_training_coordinator>
        $<TARGET_FILE:eps_training_worker> $<TARGET_FILE:eps_training_control>)
    set_tests_properties(synchronous_training_processes PROPERTIES TIMEOUT 120)
    add_executable(eps_synchronous_rpc_tests tests/synchronous_rpc_tests.cpp)
    add_executable(eps_lookup_layout_rpc_tests tests/lookup_layout_rpc_tests.cpp)
    target_link_libraries(eps_lookup_layout_rpc_tests PRIVATE eps_brpc)
    target_include_directories(eps_lookup_layout_rpc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME reusable_lookup_layout_rpc COMMAND eps_lookup_layout_rpc_tests)
    set_tests_properties(reusable_lookup_layout_rpc PROPERTIES TIMEOUT 45)
    target_link_libraries(eps_synchronous_rpc_tests PRIVATE eps_brpc)
    target_include_directories(eps_synchronous_rpc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME synchronous_gradient_rpc COMMAND eps_synchronous_rpc_tests)
    set_tests_properties(synchronous_gradient_rpc PROPERTIES TIMEOUT 30)
  add_executable(eps_gradient_layout_rpc_tests tests/gradient_layout_rpc_tests.cpp)
  target_link_libraries(eps_gradient_layout_rpc_tests PRIVATE eps_brpc)
  target_include_directories(eps_gradient_layout_rpc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME reusable_gradient_layout_rpc COMMAND eps_gradient_layout_rpc_tests)
  set_tests_properties(reusable_gradient_layout_rpc PROPERTIES TIMEOUT 30)
    add_executable(eps_worker_agent_tests tests/worker_agent_tests.cpp)
    target_link_libraries(eps_worker_agent_tests PRIVATE eps_brpc)
    add_test(NAME in_process_worker_lifecycle COMMAND eps_worker_agent_tests)
    set_tests_properties(in_process_worker_lifecycle PROPERTIES TIMEOUT 30)
    if(EPS_BUILD_CUDA)
      add_executable(eps_device_pull_cuda tests/device_pull_tests.cpp tests/device_pull_model.cu)
      target_link_libraries(eps_device_pull_cuda PRIVATE eps_brpc eps_cuda)
      target_compile_definitions(eps_device_pull_cuda PRIVATE EPS_DEVICE_CUDA=1)
      set_target_properties(eps_device_pull_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
      target_compile_options(eps_device_pull_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      add_test(NAME distributed_device_pull_cuda COMMAND eps_device_pull_cuda)
      set_tests_properties(distributed_device_pull_cuda PROPERTIES TIMEOUT 90)
      add_executable(eps_worker_agent_cuda tests/worker_agent_cuda.cu)
      target_link_libraries(eps_worker_agent_cuda PRIVATE eps_brpc eps_cuda)
      set_target_properties(eps_worker_agent_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
      target_compile_options(eps_worker_agent_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
      add_test(NAME native_worker_cuda_stream COMMAND eps_worker_agent_cuda)
      set_tests_properties(native_worker_cuda_stream PROPERTIES TIMEOUT 60)
    endif()
    add_executable(eps_agent_rpc_admission_tests tests/agent_rpc_admission_tests.cpp)
    target_link_libraries(eps_agent_rpc_admission_tests PRIVATE eps_brpc)
    target_include_directories(eps_agent_rpc_admission_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME agent_rpc_admission_queue COMMAND eps_agent_rpc_admission_tests)
    set_tests_properties(agent_rpc_admission_queue PROPERTIES TIMEOUT 60)
    add_executable(eps_shard_observation_rpc tests/shard_observation_rpc.cpp)
    target_link_libraries(eps_shard_observation_rpc PRIVATE eps_brpc)
    target_include_directories(eps_shard_observation_rpc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME shard_observation_rpc COMMAND eps_shard_observation_rpc)
    set_tests_properties(shard_observation_rpc PROPERTIES TIMEOUT 30)
    add_test(NAME agent_admission_cli
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/agent_admission_cli.py"
        $<TARGET_FILE:eps_cache_agent>)
    set_tests_properties(agent_admission_cli PROPERTIES TIMEOUT 30)
    add_executable(eps_rpc_checks tests/rpc_checks.cpp)
    target_link_libraries(eps_rpc_checks PRIVATE eps_brpc)
    target_include_directories(eps_rpc_checks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(eps_cache_policy_rpc_checks tests/cache_policy_rpc_checks.cpp)
    target_link_libraries(eps_cache_policy_rpc_checks PRIVATE eps_brpc)
    add_test(NAME automatic_cache_external_shards
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cache_policy_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_policy_rpc_checks> $<TARGET_FILE:eps_cache_agent>)
    add_test(NAME cache_policy_fixed_trace_smoke
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cache_policy_trace_processes.py"
        $<TARGET_FILE:eps_cache_policy_probe>)
    set_tests_properties(cache_policy_fixed_trace_smoke PROPERTIES TIMEOUT 150)
    add_test(NAME automatic_cache_coalesced_external_shards
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cache_policy_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_policy_rpc_checks> $<TARGET_FILE:eps_cache_agent>)
    set_tests_properties(automatic_cache_external_shards automatic_cache_coalesced_external_shards PROPERTIES TIMEOUT 120)
    set_tests_properties(automatic_cache_coalesced_external_shards PROPERTIES ENVIRONMENT "EPS_COALESCE_GATHERS=1")
    add_executable(eps_overload_checks tests/overload_checks.cpp)
    target_link_libraries(eps_overload_checks PRIVATE eps_brpc)
    add_test(NAME rpc_cache_backpressure
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/overload_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent> $<TARGET_FILE:eps_overload_checks>)
    set_tests_properties(rpc_cache_backpressure PROPERTIES TIMEOUT 60)
    add_test(NAME capacity_failed_point_diagnostics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/capacity_failure_processes.py"
        $<TARGET_FILE_DIR:eps_capacity_probe>)
    set_tests_properties(capacity_failed_point_diagnostics PROPERTIES TIMEOUT 90)
    add_test(NAME two_process_rpc_and_restore
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/rpc_processes.py"
              $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_rpc_checks>)
    set_tests_properties(two_process_rpc_and_restore PROPERTIES TIMEOUT 90)
    add_test(NAME cache_agent_process_lifecycle
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cache_rpc_processes.py"
        $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_rpc_checks> $<TARGET_FILE:eps_cache_agent>)
    set_tests_properties(cache_agent_process_lifecycle PROPERTIES TIMEOUT 90)
    if(EPS_BUILD_PYTHON)
      add_test(NAME entropy_table_hook_configuration
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_install_tests.py")
      execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import importlib.util; assert importlib.util.find_spec('prometheus_client')"
        RESULT_VARIABLE eps_prometheus_available OUTPUT_QUIET ERROR_QUIET)
      if(eps_prometheus_available EQUAL 0)
        add_test(NAME entropy_monitor_endpoint
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_metrics_endpoint.py")
        set_tests_properties(entropy_monitor_endpoint PROPERTIES TIMEOUT 15
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
      endif()
      add_test(NAME checkpoint_filesystem_contract
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/checkpoint_store_tests.py")
      add_test(NAME committed_checkpoint_staging
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/checkpoint_restore_tests.py")
      set_tests_properties(checkpoint_filesystem_contract PROPERTIES TIMEOUT 30)
      execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import importlib.util; assert importlib.util.find_spec('tensorflow')"
        RESULT_VARIABLE eps_tensorflow_available OUTPUT_QUIET ERROR_QUIET)
      if(eps_tensorflow_available EQUAL 0)
        add_test(NAME entropy_original_table_external_shards
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_hooks_processes.py" $<TARGET_FILE:eps_server>)
        add_test(NAME entropy_original_table_cache
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_hooks_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        add_test(NAME entropy_original_restore_trajectory
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_restore_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent> $<TARGET_FILE:eps_snapshot_compare>)
        add_test(NAME entropy_peer_checkpoint_control
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_peer_checkpoint.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_snapshot_compare>)
        set_tests_properties(entropy_peer_checkpoint_control PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        set_tests_properties(entropy_original_restore_trajectory PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        set_tests_properties(entropy_original_table_external_shards entropy_original_table_cache PROPERTIES
          TIMEOUT 120 SKIP_RETURN_CODE 77
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        if(EPS_BUILD_RUNTIME AND NOT APPLE)
          add_test(NAME entropy_original_table_native_cpu COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_hooks_processes.py" $<TARGET_FILE:eps_server>)
          set_tests_properties(entropy_original_table_native_cpu PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_ENTROPY_BACKEND=native;EPS_TF_OP_LIBRARY=${eps_tf_operations};EPS_NATIVE_RUNTIME=$<TARGET_FILE:eps_runtime>")
          add_test(NAME entropy_original_restore_native COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_restore_processes.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent> $<TARGET_FILE:eps_snapshot_compare>)
          set_tests_properties(entropy_original_restore_native PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_ENTROPY_BACKEND=native;EPS_TF_OP_LIBRARY=${eps_tf_operations};EPS_NATIVE_RUNTIME=$<TARGET_FILE:eps_runtime>")
          if(EPS_BUILD_CUDA)
            add_test(NAME entropy_original_table_native_cuda COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/entropy_hooks_processes.py" $<TARGET_FILE:eps_server>)
            set_tests_properties(entropy_original_table_native_cuda PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77
              ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_ENTROPY_BACKEND=native;EPS_ENTROPY_DEVICE=/GPU:0;EPS_TF_OP_LIBRARY=${eps_tf_operations};EPS_NATIVE_RUNTIME=$<TARGET_FILE:eps_runtime>")
          endif()
        endif()
        add_test(NAME tensorflow_graph_external_shards
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py" $<TARGET_FILE:eps_server>)
        add_test(NAME tensorflow_synchronous_model_cpu
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_synchronous_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-synchronous-cpu.json")
        set_tests_properties(tensorflow_synchronous_model_cpu PROPERTIES TIMEOUT 180
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_consistency_model_cpu
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_consistency_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-policy-smoke.json")
        set_tests_properties(tensorflow_consistency_model_cpu PROPERTIES TIMEOUT 180
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_owner_incarnation
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_synchronous_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --owner-incarnation replacement-2
            --out "${CMAKE_CURRENT_BINARY_DIR}/tf-owner-incarnation.json")
        set_tests_properties(tensorflow_owner_incarnation PROPERTIES TIMEOUT 180
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_joint_training_checkpoint
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_checkpoint_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-joint-checkpoint.json")
        set_tests_properties(tensorflow_joint_training_checkpoint PROPERTIES TIMEOUT 240
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_replicated_checkpoint_rollover
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_checkpoint_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --replicated-rollover
            --out "${CMAKE_CURRENT_BINARY_DIR}/tf-replicated-rollover.json")
        set_tests_properties(tensorflow_replicated_checkpoint_rollover PROPERTIES TIMEOUT 240
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_checkpoint_retirement
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_checkpoint_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --replicated-rollover --retire-old
            --out "${CMAKE_CURRENT_BINARY_DIR}/tf-checkpoint-retirement.json")
        set_tests_properties(tensorflow_checkpoint_retirement PROPERTIES TIMEOUT 240
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_repeated_retirement_recovery
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_checkpoint_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --replicated-rollover --retire-old --repeated-recovery
            --out "${CMAKE_CURRENT_BINARY_DIR}/tf-repeated-retirement-recovery.json")
        set_tests_properties(tensorflow_repeated_retirement_recovery PROPERTIES TIMEOUT 360
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_reusable_lookup_layouts
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_lookup_layouts.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-lookup-layouts.json")
        set_tests_properties(tensorflow_reusable_lookup_layouts PROPERTIES TIMEOUT 360
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_reusable_gradient_layouts
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_gradient_layouts.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-gradient-layouts.json")
        set_tests_properties(tensorflow_reusable_gradient_layouts PROPERTIES TIMEOUT 360
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_replicated_primary_recovery
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_replica_training.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-replica-recovery.json")
        set_tests_properties(tensorflow_replicated_primary_recovery PROPERTIES TIMEOUT 240
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME distributed_model_orchestration
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/distributed_model_orchestration.py"
            "${CMAKE_CURRENT_BINARY_DIR}")
        add_test(NAME durable_worker_gradients
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/worker_journal_tests.py")
        add_test(NAME worker_journal_handoff
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/journal_handoff_tests.py"
            "${CMAKE_CURRENT_BINARY_DIR}/journal-handoff-unit.json")
        set_tests_properties(worker_journal_handoff PROPERTIES TIMEOUT 120)
        add_test(NAME managed_role_processes
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/role_node_tests.py"
            "${CMAKE_CURRENT_BINARY_DIR}/managed-role-processes.json")
        set_tests_properties(managed_role_processes PROPERTIES TIMEOUT 120)
        add_test(NAME worker_gradient_replica
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gradient_replica_tests.py"
            "${CMAKE_CURRENT_BINARY_DIR}/gradient-replica-unit.json")
        set_tests_properties(worker_gradient_replica PROPERTIES TIMEOUT 120)
        add_test(NAME tensorflow_gradient_replica_failure
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_gradient_replica_failure.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/gradient-replica-failure.json")
        set_tests_properties(tensorflow_gradient_replica_failure PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME managed_gradient_replicas
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/managed_training_recovery.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --gradient-replicas
            --out "${CMAKE_CURRENT_BINARY_DIR}/gradient-replicas-model.json")
        set_tests_properties(managed_gradient_replicas PROPERTIES TIMEOUT 480
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME managed_training_recovery
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/managed_training_recovery.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/managed-recovery-model.json")
        set_tests_properties(managed_training_recovery PROPERTIES TIMEOUT 420
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_worker_journal_handoff
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_worker_recovery.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --owners host --handoff
            --out "${CMAKE_CURRENT_BINARY_DIR}/tf-journal-handoff.json")
        set_tests_properties(tensorflow_worker_journal_handoff PROPERTIES TIMEOUT 360
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_worker_recovery
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_worker_recovery.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-worker-recovery.json")
        set_tests_properties(tensorflow_worker_recovery PROPERTIES TIMEOUT 300
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_read_failures
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_read_failures.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --out "${CMAKE_CURRENT_BINARY_DIR}/tf-read-failures.json")
        set_tests_properties(tensorflow_read_failures PROPERTIES TIMEOUT 240
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        add_test(NAME tensorflow_replica_owner_recovery_host
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_worker_recovery.py"
            --build "${CMAKE_CURRENT_BINARY_DIR}" --owners host --out "${CMAKE_CURRENT_BINARY_DIR}/tf-owner-recovery-host.json")
        set_tests_properties(tensorflow_replica_owner_recovery_host PROPERTIES TIMEOUT 300
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        set_tests_properties(durable_worker_gradients PROPERTIES TIMEOUT 30)
        set_tests_properties(distributed_model_orchestration PROPERTIES TIMEOUT 120)
        if(EPS_BUILD_CUDA)
          add_test(NAME tensorflow_repeated_retirement_recovery_cuda
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_checkpoint_training.py"
              --build "${CMAKE_CURRENT_BINARY_DIR}" --replicated-rollover --retire-old --repeated-recovery --gpu
              --out "${CMAKE_CURRENT_BINARY_DIR}/tf-repeated-retirement-recovery-cuda.json")
          set_tests_properties(tensorflow_repeated_retirement_recovery_cuda PROPERTIES TIMEOUT 480
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          add_test(NAME tensorflow_read_failures_cuda
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_read_failures.py"
              --build "${CMAKE_CURRENT_BINARY_DIR}" --cuda --out "${CMAKE_CURRENT_BINARY_DIR}/tf-read-failures-cuda.json")
          set_tests_properties(tensorflow_read_failures_cuda PROPERTIES TIMEOUT 300
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          add_test(NAME tensorflow_synchronous_model_cuda
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_synchronous_training.py"
              --build "${CMAKE_CURRENT_BINARY_DIR}" --devices 0 --out "${CMAKE_CURRENT_BINARY_DIR}/tf-synchronous-cuda.json")
          set_tests_properties(tensorflow_synchronous_model_cuda PROPERTIES TIMEOUT 240
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
          add_test(NAME tensorflow_replica_owner_recovery_cuda
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_worker_recovery.py"
              --build "${CMAKE_CURRENT_BINARY_DIR}" --owners cuda --out "${CMAKE_CURRENT_BINARY_DIR}/tf-owner-recovery-cuda.json")
          set_tests_properties(tensorflow_replica_owner_recovery_cuda PROPERTIES TIMEOUT 360
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        endif()
        add_test(NAME tensorflow_graph_cache_external_shards
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        add_test(NAME tensorflow_graph_automatic_cache
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        add_test(NAME tensorflow_graph_automatic_coalesced_cache
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        set_tests_properties(tensorflow_graph_external_shards tensorflow_graph_cache_external_shards PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy")
        set_tests_properties(tensorflow_graph_automatic_cache PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_AUTO_CACHE=1")
        set_tests_properties(tensorflow_graph_automatic_coalesced_cache PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_AUTO_CACHE=1;EPS_TEST_COALESCE=1")
        add_test(NAME tensorflow_graph_prepared_external_shards
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py" $<TARGET_FILE:eps_server>)
        set_tests_properties(tensorflow_graph_prepared_external_shards PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_PREPARE=1")
        add_test(NAME tensorflow_graph_prepared_automatic_cache
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py" $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        set_tests_properties(tensorflow_graph_prepared_automatic_cache PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_PREPARE=1;EPS_TEST_AUTO_CACHE=1;EPS_TEST_COALESCE=1")
        add_test(NAME tensorflow_graph_coalesced_cache
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        set_tests_properties(tensorflow_graph_coalesced_cache PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_COALESCE=1")
        add_test(NAME tensorflow_graph_periodic_checkpoint
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensorflow_processes.py"
            $<TARGET_FILE:eps_server> $<TARGET_FILE:eps_cache_agent>)
        set_tests_properties(tensorflow_graph_periodic_checkpoint PROPERTIES TIMEOUT 120
          ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python:${CMAKE_CURRENT_SOURCE_DIR}/adapters/entropy;EPS_TEST_PREPARE=1;EPS_TEST_AUTO_CACHE=1;EPS_TEST_PERIODIC=1")
      endif()
    endif()
    if(EPS_BUILD_CUDA)
      set_tests_properties(cache_agent_process_lifecycle PROPERTIES ENVIRONMENT "EPS_CACHE_BACKEND=cuda")
      set_tests_properties(rpc_cache_backpressure PROPERTIES ENVIRONMENT "EPS_CACHE_BACKEND=cuda")
    endif()
  endif()
endif()
if(EPS_BUILD_TESTS)
  enable_testing()
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME original_native_source_bundle
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/original_native_bundle_tests.py")
    set_tests_properties(original_native_source_bundle PROPERTIES TIMEOUT 30)
  endif()
  add_executable(eps_device_pull_tests tests/device_pull_tests.cpp)
  target_link_libraries(eps_device_pull_tests PRIVATE eps_core)
  add_test(NAME distributed_device_pull_protocol COMMAND eps_device_pull_tests)
  set_tests_properties(distributed_device_pull_protocol PROPERTIES TIMEOUT 30)
  add_executable(eps_agent_admission_tests tests/agent_admission_tests.cpp)
  target_link_libraries(eps_agent_admission_tests PRIVATE eps_core)
  target_include_directories(eps_agent_admission_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME agent_admission_lifecycle COMMAND eps_agent_admission_tests)
  set_tests_properties(agent_admission_lifecycle PROPERTIES TIMEOUT 30)
  add_executable(eps_core_tests tests/core_tests.cpp)
  target_link_libraries(eps_core_tests PRIVATE eps_core)
  add_test(NAME core_correctness COMMAND eps_core_tests)
  add_executable(eps_planner_tests tests/planner_tests.cpp)
  target_link_libraries(eps_planner_tests PRIVATE eps_core)
  add_test(NAME cost_and_migration_policies COMMAND eps_planner_tests)
  add_executable(eps_cache_tests tests/cache_tests.cpp)
  target_link_libraries(eps_cache_tests PRIVATE eps_core)
  add_test(NAME cache_ownership_and_updates COMMAND eps_cache_tests)
  add_executable(eps_cache_policy_tests tests/cache_policy_tests.cpp)
  target_link_libraries(eps_cache_policy_tests PRIVATE eps_core)
  target_include_directories(eps_cache_policy_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME automatic_cache_policy_lifecycle COMMAND eps_cache_policy_tests)
  set_tests_properties(automatic_cache_policy_lifecycle PROPERTIES TIMEOUT 30)
  add_executable(eps_agent_tests tests/agent_tests.cpp)
  target_link_libraries(eps_agent_tests PRIVATE eps_core)
  add_test(NAME agent_control_replay COMMAND eps_agent_tests)
  add_executable(eps_adapter_tests tests/adapter_tests.cpp)
  target_link_libraries(eps_adapter_tests PRIVATE eps_core)
  add_test(NAME embedding_adapter_lifecycle COMMAND eps_adapter_tests)
  add_executable(eps_read_cleanup_tests tests/read_cleanup_tests.cpp)
  target_link_libraries(eps_read_cleanup_tests PRIVATE eps_core)
  add_test(NAME read_lifecycle_cleanup COMMAND eps_read_cleanup_tests)
  set_tests_properties(read_lifecycle_cleanup PROPERTIES TIMEOUT 60)
  add_executable(eps_phase_tests tests/phase_timing_tests.cpp)
  target_link_libraries(eps_phase_tests PRIVATE eps_core)
  add_test(NAME service_phase_timing COMMAND eps_phase_tests)
  add_executable(eps_preparation_tests tests/preparation_tests.cpp)
  target_link_libraries(eps_preparation_tests PRIVATE eps_core)
  target_include_directories(eps_preparation_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME bounded_asynchronous_preparation COMMAND eps_preparation_tests)
  add_executable(eps_periodic_checkpoint_tests tests/periodic_checkpoint_tests.cpp)
  target_link_libraries(eps_periodic_checkpoint_tests PRIVATE eps_core)
  add_test(NAME periodic_checkpoint_lifecycle COMMAND eps_periodic_checkpoint_tests)
  set_tests_properties(periodic_checkpoint_lifecycle PROPERTIES TIMEOUT 30)
  add_executable(eps_training_progress_tests tests/training_progress_tests.cpp)
  target_link_libraries(eps_training_progress_tests PRIVATE eps_core)
  add_test(NAME training_progress_consistency COMMAND eps_training_progress_tests)
  set_tests_properties(training_progress_consistency PROPERTIES TIMEOUT 30)
  add_executable(eps_synchronous_gradients_tests tests/synchronous_gradients_tests.cpp)
  add_executable(eps_lookup_layout_cache_tests tests/lookup_layout_cache_tests.cpp)
  add_executable(eps_replica_log_tests tests/replica_log_tests.cpp)
  target_link_libraries(eps_replica_log_tests PRIVATE eps_core)
  add_test(NAME durable_replica_log COMMAND eps_replica_log_tests)
  target_link_libraries(eps_lookup_layout_cache_tests PRIVATE eps_core)
  add_test(NAME reusable_lookup_layout_cache COMMAND eps_lookup_layout_cache_tests)
  add_executable(eps_gradient_layout_cache_tests tests/gradient_layout_cache_tests.cpp)
  target_link_libraries(eps_gradient_layout_cache_tests PRIVATE eps_core)
  add_test(NAME reusable_gradient_layout_cache COMMAND eps_gradient_layout_cache_tests)
  target_link_libraries(eps_synchronous_gradients_tests PRIVATE eps_core)
  add_test(NAME synchronous_gradient_rounds COMMAND eps_synchronous_gradients_tests)
  set_tests_properties(synchronous_gradient_rounds PROPERTIES TIMEOUT 30)
  add_executable(eps_weight_bounds_tests tests/weight_bounds_tests.cpp)
  add_executable(eps_restore_provenance_tests tests/restore_provenance_tests.cpp)
  target_link_libraries(eps_restore_provenance_tests PRIVATE eps_core)
  target_include_directories(eps_restore_provenance_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME restored_snapshot_provenance COMMAND eps_restore_provenance_tests)
  add_executable(eps_snapshot_compare tests/snapshot_compare.cpp)
  target_link_libraries(eps_snapshot_compare PRIVATE eps_core)
  target_include_directories(eps_snapshot_compare PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(eps_weight_bounds_tests PRIVATE eps_core)
  add_test(NAME bounded_sparse_optimizer COMMAND eps_weight_bounds_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/snapshot-v1-unbounded.bin")
  set_tests_properties(bounded_sparse_optimizer PROPERTIES TIMEOUT 30)
  add_executable(eps_resumable_actor_tests tests/resumable_actor_tests.cpp)
  target_link_libraries(eps_resumable_actor_tests PRIVATE eps_core)
  target_include_directories(eps_resumable_actor_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME resumable_actor_dependencies COMMAND eps_resumable_actor_tests)
  set_tests_properties(resumable_actor_dependencies PROPERTIES TIMEOUT 30)
  add_executable(eps_resumable_service_tests tests/resumable_service_tests.cpp)
  target_link_libraries(eps_resumable_service_tests PRIVATE eps_core)
  add_test(NAME resumable_shard_operations COMMAND eps_resumable_service_tests)
  set_tests_properties(resumable_shard_operations PROPERTIES TIMEOUT 45)
  set_tests_properties(bounded_asynchronous_preparation PROPERTIES TIMEOUT 30)
  add_executable(eps_multi_agent_tests tests/multi_agent_tests.cpp)
  target_link_libraries(eps_multi_agent_tests PRIVATE eps_core)
  add_test(NAME concurrent_multi_agent_updates COMMAND eps_multi_agent_tests)
  set_tests_properties(concurrent_multi_agent_updates PROPERTIES TIMEOUT 15)
  add_executable(eps_coalesced_gather_tests tests/coalesced_gather_tests.cpp)
  target_link_libraries(eps_coalesced_gather_tests PRIVATE eps_core)
  add_test(NAME coalesced_cross_shard_gathers COMMAND eps_coalesced_gather_tests)
  set_tests_properties(coalesced_cross_shard_gathers PROPERTIES TIMEOUT 20)
  add_executable(eps_compact_tests tests/compact_tests.cpp)
  target_link_libraries(eps_compact_tests PRIVATE eps_core)
  target_include_directories(eps_compact_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME compact_feature_gradient_semantics COMMAND eps_compact_tests)
  if(EPS_BUILD_CUDA)
    target_link_libraries(eps_resumable_service_tests PRIVATE eps_cuda)
    target_compile_definitions(eps_resumable_service_tests PRIVATE EPS_HAS_CUDA=1)
    add_test(NAME resumable_cuda_shard_operations COMMAND eps_resumable_service_tests)
    set_tests_properties(resumable_cuda_shard_operations PROPERTIES TIMEOUT 90
      ENVIRONMENT "EPS_RESUMABLE_BACKEND=cuda")
    add_executable(eps_cuda_tests tests/cuda_tests.cpp)
    target_link_libraries(eps_cuda_tests PRIVATE eps_cuda)
    add_test(NAME cuda_state_and_device_results COMMAND eps_cuda_tests)
    set_tests_properties(cuda_state_and_device_results PROPERTIES TIMEOUT 60)
    add_executable(eps_cuda_stress tests/cuda_stress.cpp)
    target_link_libraries(eps_cuda_stress PRIVATE eps_cuda)
    add_test(NAME cuda_concurrent_checked_reads COMMAND eps_cuda_stress)
    set_tests_properties(cuda_concurrent_checked_reads PROPERTIES TIMEOUT 120)
  endif()
endif()

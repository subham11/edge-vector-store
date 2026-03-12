# ──────────────────────────────────────────────────────────────
#  EdgeVectorStore — C++ build (shared by Android + WASM)
# ──────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.13)
project(EdgeVectorStore LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── sources ──────────────────────────────────────────────────

set(EVS_SOURCES
    cpp/core/Journal.cpp
    cpp/core/EdgeStore.cpp
    cpp/storage/MetadataStore.cpp
    cpp/storage/MmapFile.cpp
    cpp/storage/VectorStore.cpp
    cpp/ann/ANNEngine.cpp
    cpp/ann/HotCache.cpp
    cpp/ann/TieredSearch.cpp
    cpp/ann/SIMDKernels.cpp
    cpp/pack/PackFormat.cpp
    cpp/pack/PackIO.cpp
    cpp/bench/BenchmarkEngine.cpp
    # USearch C API (compiled in-tree)
    third_party/usearch/c/lib.cpp
)

# Suppress USearch's half-precision #warning (vendored code, intentionally standalone)
set_source_files_properties(third_party/usearch/c/lib.cpp PROPERTIES
    COMPILE_FLAGS "-Wno-#warnings"
)

set(EVS_SOURCES ${EVS_SOURCES}
    # miniz (ZIP I/O)
    third_party/miniz/miniz.c
    third_party/miniz/miniz_tdef.c
    third_party/miniz/miniz_tinfl.c
    third_party/miniz/miniz_zip.c
)

set(EVS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/usearch/c
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniz
)

# ── USearch config: standalone, no optional deps ─────────────

add_definitions(
    -DUSEARCH_USE_SIMSIMD=0
    -DUSEARCH_USE_FP16LIB=0
    -DUSEARCH_USE_OPENMP=0
)

# ══════════════════════════════════════════════════════════════
#  Target: Desktop test (no JSI, static lib + test main)
# ══════════════════════════════════════════════════════════════

if(EVS_DESKTOP_TEST)
    add_library(evs_core STATIC ${EVS_SOURCES})
    target_include_directories(evs_core PUBLIC ${EVS_INCLUDE_DIRS})
    target_compile_options(evs_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

    # Smoke-test executable
    add_executable(evs_test test/smoke_test.cpp)
    target_link_libraries(evs_test evs_core)
    target_include_directories(evs_test PRIVATE ${EVS_INCLUDE_DIRS})

    # Benchmark executable (100K vector benchmark)
    if(EVS_BENCHMARK)
        add_executable(evs_bench bench/benchmark.cpp)
        target_link_libraries(evs_bench evs_core)
        target_include_directories(evs_bench PRIVATE
            ${EVS_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/bench
        )
    endif()

# ══════════════════════════════════════════════════════════════
#  Target: React Native (Android / iOS shared library)
# ══════════════════════════════════════════════════════════════

elseif(NOT EVS_WASM)
    # On Android this is invoked by RN's CMake integration.
    # On iOS the podspec compiles the same sources directly.
    list(APPEND EVS_SOURCES cpp/bridge/EdgeStoreModule.cpp)

    add_library(edge_vector_store SHARED ${EVS_SOURCES})

    target_include_directories(edge_vector_store PRIVATE
        ${EVS_INCLUDE_DIRS}
    )

    # React Native JSI headers (provided by the RN build system)
    if(DEFINED REACT_NATIVE_DIR)
        target_include_directories(edge_vector_store PRIVATE
            ${REACT_NATIVE_DIR}/ReactCommon
            ${REACT_NATIVE_DIR}/ReactCommon/jsi
        )
    endif()

    if(ANDROID)
        target_link_libraries(edge_vector_store log)
    endif()

# ══════════════════════════════════════════════════════════════
#  Target: WASM (Emscripten)
# ══════════════════════════════════════════════════════════════

else()
    # Build a .wasm module with embind exports.
    # No JSI bridge — JS calls the C++ directly via embind.
    add_executable(evs_wasm ${EVS_SOURCES})

    target_include_directories(evs_wasm PRIVATE ${EVS_INCLUDE_DIRS})

    set_target_properties(evs_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -s MODULARIZE=1 \
            -s EXPORT_NAME='createEVS' \
            -s ALLOW_MEMORY_GROWTH=1 \
            -s FILESYSTEM=1 \
            -s EXPORTED_RUNTIME_METHODS=['FS','callMain'] \
            --bind"
    )
endif()

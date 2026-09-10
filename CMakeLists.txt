cmake_minimum_required(VERSION 3.19)
set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
project(tts-cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(TTS_CPP_BUILD_EXECUTABLES "" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)
if(NOT GGML_VULKAN OR GGML_CUDA OR GGML_CPU)
    message(FATAL_ERROR "tts-cpp requires GGML_VULKAN=ON, GGML_CUDA=OFF, GGML_CPU=OFF")
endif()
add_library(tts-cpp STATIC
    src/main.cpp
    src/t3_nano.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp)
target_include_directories(tts-cpp PUBLIC include PRIVATE src ggml/include)
target_link_libraries(tts-cpp PUBLIC ggml)
target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
if(MSVC)
    target_compile_definitions(tts-cpp PRIVATE _USE_MATH_DEFINES)
    target_compile_options(tts-cpp PRIVATE /O2 /DNDEBUG)
endif()
set_target_properties(tts-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TTS_CPP_BUILD_EXECUTABLES)
    add_executable(chatterbox-server src/server.cpp)
    set_target_properties(chatterbox-server PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_compile_definitions(chatterbox-server PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    if(MSVC)
        target_compile_options(chatterbox-server PRIVATE /MP /bigobj /O2 /DNDEBUG)
    endif()
    target_link_libraries(chatterbox-server PRIVATE tts-cpp)
endif()

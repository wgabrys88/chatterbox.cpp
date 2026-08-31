cmake_minimum_required(VERSION 3.19)
project(tts-cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(TTS_CPP_BUILD_EXECUTABLES "" OFF)
option(TTS_CPP_BUILD_TESTS "" OFF)
add_subdirectory(ggml)
if(NOT GGML_VULKAN OR GGML_CUDA)
    message(FATAL_ERROR "tts-cpp bare-metal fork requires GGML_VULKAN=ON and GGML_CUDA=OFF")
endif()
find_package(OpenMP)
add_library(mtl_tokenizer STATIC src/mtl_tokenizer.cpp)
target_include_directories(mtl_tokenizer PUBLIC src)
add_library(tts-cpp STATIC
    src/main.cpp
    src/gpt2_bpe.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/mel_extract_stft.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
    src/t3_mtl.cpp)
target_include_directories(tts-cpp PUBLIC include PRIVATE src ggml/include)
target_link_libraries(tts-cpp PUBLIC ggml PRIVATE mtl_tokenizer)
target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
if(MSVC)
    target_compile_definitions(tts-cpp PRIVATE _USE_MATH_DEFINES)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(tts-cpp PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(tts-cpp mtl_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

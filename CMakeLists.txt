cmake_minimum_required(VERSION 3.19)
project(tts-cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(ggml)
if(NOT GGML_VULKAN)
    message(FATAL_ERROR "tts-cpp bare-metal fork requires GGML_VULKAN=ON")
endif()
find_package(OpenMP REQUIRED)
add_library(tts-cpp STATIC
    src/main.cpp
    src/chatterbox_tts.cpp
    src/chatterbox_engine.cpp
    src/voice_features.cpp
    src/voice_encoder.cpp
    src/campplus.cpp
    src/s3tokenizer.cpp
    src/t3_mtl.cpp
    src/mtl_tokenizer.cpp)
target_include_directories(tts-cpp PUBLIC include PRIVATE src ggml/include)
target_link_libraries(tts-cpp PRIVATE ggml OpenMP::OpenMP_CXX)
target_compile_definitions(tts-cpp PRIVATE GGML_USE_VULKAN)
if(MSVC)
    target_compile_definitions(tts-cpp PRIVATE _USE_MATH_DEFINES)
endif()

cmake_minimum_required(VERSION 3.19)
set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
project(tts-cpp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(TTS_CPP_BUILD_EXECUTABLES "" OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(TTS_FAMILY "gpt2" CACHE STRING "gpt2 or v3")
set_property(CACHE TTS_FAMILY PROPERTY STRINGS gpt2 v3)
if("${TTS_FAMILY}" STREQUAL "v3")
    set(TTS_FAMILY_DEFINE TTS_FAMILY_V3)
    set(TTS_T3_SOURCES
        src/t3_v3.cpp
        src/mtl_numbers.cpp
        src/mtl_external_tokenizer.cpp
)
    set(TTS_SYSTEM_LIBS icu)
elseif("${TTS_FAMILY}" STREQUAL "gpt2")
    set(TTS_FAMILY_DEFINE TTS_FAMILY_GPT2)
    set(TTS_T3_SOURCES
        src/t3_gpt2.cpp
        src/gpt2_bpe.cpp
)
else()
    message(FATAL_ERROR "TTS_FAMILY must be gpt2 or v3 (got '${TTS_FAMILY}')")
endif()
find_package(Vulkan REQUIRED)
add_subdirectory(ggml)
add_compile_definitions(${TTS_FAMILY_DEFINE})
add_library(tts-cpp STATIC ${TTS_T3_SOURCES} src/text_prepare.cpp src/chatterbox_s3.cpp src/chatterbox_engine.cpp)
target_include_directories(tts-cpp PUBLIC include PRIVATE src ggml/include)
target_link_libraries(tts-cpp PUBLIC ggml Vulkan::Vulkan ${TTS_SYSTEM_LIBS})
function(configure_tts target)
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN GGML_USE_VULKAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    if(MSVC)
        target_compile_options(${target} PRIVATE /bigobj /O2 /DNDEBUG)
    endif()
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
endfunction()
configure_tts(tts-cpp)
if(TTS_CPP_BUILD_EXECUTABLES)
    add_library(tts-cpp-bake STATIC
        src/main.cpp
        src/voice_features.cpp
        src/mel_extract_stft.cpp
        src/voice_encoder.cpp
        src/campplus.cpp
        src/s3tokenizer.cpp)
    target_include_directories(tts-cpp-bake PUBLIC include PRIVATE src ggml/include)
    target_link_libraries(tts-cpp-bake PUBLIC ggml)
    configure_tts(tts-cpp-bake)
    add_executable(chatterbox-server src/server.cpp)
    configure_tts(chatterbox-server)
    target_link_libraries(chatterbox-server PRIVATE tts-cpp)
    add_executable(chatterbox-bake src/bake.cpp)
    configure_tts(chatterbox-bake)
    target_include_directories(chatterbox-bake PRIVATE src ggml/include)
    target_link_libraries(chatterbox-bake PRIVATE tts-cpp-bake Vulkan::Vulkan)
endif()

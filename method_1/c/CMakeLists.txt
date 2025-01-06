cmake_minimum_required(VERSION 3.10)

# Project name and version
project(Skeletonization VERSION 1.0)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add source files
set(SOURCES
    src/main.c
    src/skeletongrad.c
)

# Add header files
set(HEADERS
    src/skeletongrad.h
    stb_image.h
    stb_image_write.h
)

# Add executable
add_executable(skelgrad ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(skelgrad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
if (WIN32)
    if (MSVC)
        target_link_libraries(skelgrad PRIVATE ws2_32)
    elseif (MINGW)
        target_link_libraries(skelgrad PRIVATE ws2_32)
    endif()
elseif (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    target_link_libraries(skelgrad PRIVATE ${COREFOUNDATION_LIBRARY})
elseif (UNIX)
    target_link_libraries(skelgrad PRIVATE m)
endif()

# Compiler options
if (MSVC)
    target_compile_options(skelgrad PRIVATE /W4 /WX)
else()
    target_compile_options(skelgrad PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
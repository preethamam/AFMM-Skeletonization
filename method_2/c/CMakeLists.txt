cmake_minimum_required(VERSION 3.10)

# Project name
project(AFMM)

# Set the C standard
set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

# Source files
set(SOURCES
    src/main.c
    src/afmm.c
)

# Executable
add_executable(afmm ${SOURCES})

# Platform-specific settings
if(WIN32)
    if(MSVC)
        target_compile_definitions(afmm PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_link_libraries(afmm ws2_32)
    elseif(MINGW)
        target_link_libraries(afmm ws2_32)
    endif()
elseif(APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(afmm Threads::Threads)
elseif(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(afmm Threads::Threads)
endif()

# Output directories
set_target_properties(afmm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
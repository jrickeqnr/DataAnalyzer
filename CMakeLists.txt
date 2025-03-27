cmake_minimum_required(VERSION 3.14)
project(DataAnalyzer VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
if(MSVC)
    add_compile_options(/W4 /EHsc /bigobj)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Add source files
set(SOURCES
    src/main.cpp
    src/data_handler.cpp
    src/gui.cpp
    src/gui_screens.cpp  # Separate file for screen implementations
    src/plotting.cpp
    # Model implementations
    src/models/linear_regression.cpp
    src/models/elasticnet.cpp
    src/models/xgboost.cpp
    src/models/gradient_boosting.cpp
    src/models/neural_network.cpp
)

# Add header files
set(HEADERS
    include/model.h
    include/data_handler.h
    include/gui.h
    include/plotting.h
)

# Build the executable
add_executable(DataAnalyzer ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(DataAnalyzer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/implot
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/glfw/deps
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/fast-cpp-csv-parser
)

# Deal with platform-specific libraries
if(APPLE)
    # macOS-specific settings
    target_link_libraries(DataAnalyzer PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreFoundation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGL_SILENCE_DEPRECATION")
elseif(UNIX AND NOT APPLE)
    # Linux-specific settings
    find_package(X11 REQUIRED)
    target_link_libraries(DataAnalyzer PRIVATE X11 dl pthread)
elseif(WIN32)
    # Windows-specific settings
    target_compile_definitions(DataAnalyzer PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX  # Prevents Windows' min/max macros from conflicting with std::min/max
        WIN32_LEAN_AND_MEAN  # Excludes rarely-used Windows headers
    )
endif()

# Set OpenGL libraries
find_package(OpenGL REQUIRED)

# Add ImGui sources
set(IMGUI_SOURCES
    lib/imgui/imgui.cpp
    lib/imgui/imgui_demo.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/imgui_tables.cpp
    lib/imgui/imgui_widgets.cpp
    lib/imgui/backends/imgui_impl_glfw.cpp
    lib/imgui/backends/imgui_impl_opengl3.cpp
)

# Add ImPlot sources
set(IMPLOT_SOURCES
    lib/implot/implot.cpp
    lib/implot/implot_items.cpp
)

# Add sources to the executable
target_sources(DataAnalyzer PRIVATE ${IMGUI_SOURCES} ${IMPLOT_SOURCES})

# Build GLFW from source
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/glfw)

# Link libraries
target_link_libraries(DataAnalyzer PRIVATE
    OpenGL::GL
    glfw
)

# Installation
install(TARGETS DataAnalyzer DESTINATION bin)

# Create a sample data directory for testing
install(DIRECTORY DESTINATION ${CMAKE_INSTALL_PREFIX}/share/DataAnalyzer/data)

# Copy README
install(FILES README.md DESTINATION ${CMAKE_INSTALL_PREFIX}/share/DataAnalyzer)

# Output directory
set_target_properties(DataAnalyzer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Enable testing
enable_testing()

# Print information about the build
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}") 
cmake_minimum_required(VERSION 3.16)
project(SmartPPE_Safwan_DeSilva CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# 1. SET PATHS TO YOUR LOCAL INSTALLATIONS (Update these if paths differ)
# ==============================================================================
# Path to your OpenCV build folder (where OpenCVConfig.cmake resides)
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

# Path to your ONNX Runtime C++ release folder (unzipped)
set(ONNXRUNTIME_DIR "C:/onnxruntime") 
# ==============================================================================

# Locate OpenCV
find_package(OpenCV REQUIRED)

# Include Directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

# Executable target
add_executable(SmartPPE src/main.cpp)

# Link Libraries
target_link_libraries(SmartPPE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
)
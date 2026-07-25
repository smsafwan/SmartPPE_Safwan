cmake_minimum_required(VERSION 3.15)
project(SmartPPE_Safwan_DeSilva CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bypass the root version-checker by pointing directly to the vc16 library configuration
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Define ONNX Runtime Paths
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/external/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# Compile the Executable
add_executable(smart_ppe src/main.cpp)

# Link Libraries
target_link_libraries(smart_ppe PRIVATE ${OpenCV_LIBS} )
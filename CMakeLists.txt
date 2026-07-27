cmake_minimum_required(VERSION 3.15)
project(SmartPPE_Safwan_DeSilva CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Point directly to your OpenCV installation (Verify if E: or C:)
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# 2. Define ONNX Runtime Paths
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/external/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# 3. Compile the Executable
add_executable(smart_ppe src/main.cpp)

# 4. Attach Include Directories & Link Libraries directly to smart_ppe
target_include_directories(smart_ppe PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(smart_ppe PRIVATE ${OpenCV_LIBS})



# --- 3. Diagnostics Executable ---
add_executable(test_env src/test_env.cpp)
target_include_directories(test_env PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_env PRIVATE ${OpenCV_LIBS} onnxruntime)
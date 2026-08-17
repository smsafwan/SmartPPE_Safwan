cmake_minimum_required(VERSION 3.15)
project(SmartPPE_Safwan_DeSilva CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Point directly to your OpenCV installation (Verify if E: or C:)
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

# --------------------------------------------------------------------
# 2. Configure ONNX Runtime Path (UPDATE THIS TO YOUR ACTUAL PATH)
# --------------------------------------------------------------------
set(ONNXRUNTIME_DIR "C:/onnxruntime" CACHE PATH "Path to ONNX Runtime")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# --------------------------------------------------------------------
# 3. Define Executable
# --------------------------------------------------------------------
add_executable(SmartPPE_App
    src/main.cpp
)

# --------------------------------------------------------------------
# 4. Link Libraries
# --------------------------------------------------------------------
target_link_libraries(SmartPPE_App PRIVATE 
    ${OpenCV_LIBS}
    onnxruntime
)
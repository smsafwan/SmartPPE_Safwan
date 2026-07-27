#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      System Infrastructure Test        " << std::endl;
    std::cout << "========================================\n" << std::endl;

    // --- 1. OpenCV Verification ---
    std::cout << "[1] Testing OpenCV Integration..." << std::endl;
    std::cout << "    - Version: " << CV_VERSION << std::endl;
    
    cv::Mat test_mat = cv::Mat::zeros(100, 100, CV_8UC3);
    if (!test_mat.empty()) {
        std::cout << "    - Status: SUCCESS (Matrix allocated)" << std::endl;
    } else {
        std::cout << "    - Status: FAILED (Matrix empty)" << std::endl;
    }
    std::cout << std::endl;

    // --- 2. ONNX Runtime & CUDA Verification ---
    std::cout << "[2] Testing ONNX Runtime & Hardware Acceleration..." << std::endl;
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HardwareTest");
        std::cout << "    - Status: SUCCESS (Environment initialized)" << std::endl;

        // Query available Execution Providers (EPs)
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        std::cout << "    - Available Execution Providers: ";
        bool has_cuda = false;
        
        for (const auto& p : providers) {
            std::cout << p << " ";
            if (p == "CUDAExecutionProvider") {
                has_cuda = true;
            }
        }
        std::cout << "\n";

        if (has_cuda) {
            std::cout << "    - GPU Acceleration: SUCCESS (CUDA Engine Active)" << std::endl;
        } else {
            std::cout << "    - GPU Acceleration: FAILED (Falling back to CPU)" << std::endl;
            std::cout << "      Ensure CUDA and cuDNN .dll files are alongside the executable." << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cout << "    - ONNX Status: FAILED" << std::endl;
        std::cout << "    - Error Code: " << e.what() << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Press Enter to exit the diagnostic tool..." << std::endl;
    std::cin.get();
    
    return 0;
}
#include <iostream>

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define PPE_OPENCV_VERSION CV_VERSION
#else
#define PPE_OPENCV_VERSION "OpenCV not found"
#endif

int main() {
    std::cout << "PPE Project Infrastructure Initialized!" << std::endl;
    std::cout << "OpenCV Version: " << PPE_OPENCV_VERSION << std::endl;
    
    std::cout << "\nPress Enter to close the program..." << std::endl;
    std::cin.get(); // Forces the program to wait for user input
    
    return 0;
}
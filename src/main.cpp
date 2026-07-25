#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "PPE Project Infrastructure Initialized!" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    
    std::cout << "\nPress Enter to close the program..." << std::endl;
    std::cin.get(); // Forces the program to wait for user input
    
    return 0;
}
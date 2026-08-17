#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Configuration
const std::string MODEL_PATH = "C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px-2/weights/best.onnx";
const int INPUT_WIDTH = 800;
const int INPUT_HEIGHT = 800;

const std::vector<std::string> CLASS_NAMES = {"Bare_Head", "Hat_Not_OK", "NO_Safety_vest", "Safety_vest_OK", "helmet_OK"};
const std::vector<cv::Scalar> CLASS_COLORS = {cv::Scalar(0,0,255), cv::Scalar(0,165,255), cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(0,255,0)};
const std::vector<float> CLASS_THRESHOLDS = {0.50f, 0.50f, 0.50f, 0.55f, 0.55f};

std::vector<float> preprocess_image(const cv::Mat& frame, float& scale_x, float& scale_y) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    scale_x = static_cast<float>(frame.cols) / INPUT_WIDTH;
    scale_y = static_cast<float>(frame.rows) / INPUT_HEIGHT;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_tensor_values(1 * 3 * INPUT_HEIGHT * INPUT_WIDTH);
    std::vector<cv::Mat> chw_channels(3);
    for (int i = 0; i < 3; ++i) {
        chw_channels[i] = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_32FC1, input_tensor_values.data() + i * INPUT_HEIGHT * INPUT_WIDTH);
    }
    cv::split(rgb, chw_channels);
    return input_tensor_values;
}

int main(int argc, char** argv) {
    std::cout << "===========================================\n";
    std::cout << "   Smart PPE C++ Engine (DEBUG MODE)       \n";
    std::cout << "===========================================\n\n";

    try {
        std::cout << "[DEBUG 1] Initializing ONNX Environment...\n";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SmartPPE_CPP");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "[DEBUG 2] Attempting to append CUDA Provider...\n";
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[SUCCESS] CUDA Execution Provider linked.\n";

        std::cout << "[DEBUG 3] Loading model into GPU VRAM (This may take a moment)...\n";
        std::wstring widestr = std::wstring(MODEL_PATH.begin(), MODEL_PATH.end());
        Ort::Session session(env, widestr.c_str(), session_options);
        std::cout << "[SUCCESS] Model loaded successfully.\n";

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

        std::string video_source = (argc > 1) ? argv[1] : "0";
        std::cout << "[DEBUG 4] Opening video source: " << video_source << "\n";
        cv::VideoCapture cap;
        if (video_source == "0") cap.open(0);
        else cap.open(video_source);

        if (!cap.isOpened()) {
            std::cerr << "[FATAL] Could not open video source.\n";
            return -1;
        }

        cv::Mat frame;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::cout << "[DEBUG 5] Entering video loop...\n";
        int frame_count = 0;

        while (cap.read(frame)) {
            if (frame.empty()) {
                std::cout << "[INFO] Video stream ended.\n";
                break;
            }
            
            if (frame_count == 0) std::cout << "[DEBUG 6] First frame captured successfully.\n";

            float scale_x = 1.0f, scale_y = 1.0f;
            std::vector<float> input_tensor_values = preprocess_image(frame, scale_x, scale_y);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size());

            if (frame_count == 0) std::cout << "[DEBUG 7] Executing first GPU inference...\n";
            
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names, &input_tensor, 1, 
                output_names, 1);

            if (frame_count == 0) std::cout << "[DEBUG 8] Inference succeeded. Parsing tensor...\n";

            float* float_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            int num_detections = static_cast<int>(output_shape[1]);
            int num_features = static_cast<int>(output_shape[2]);

            for (int i = 0; i < num_detections; ++i) {
                float* row = float_data + i * num_features;
                float confidence = row[4];
                int class_id = static_cast<int>(row[5]);

                if (class_id < 0 || class_id >= static_cast<int>(CLASS_THRESHOLDS.size())) continue;
                if (confidence < CLASS_THRESHOLDS[class_id]) continue;

                float cx = row[0] * INPUT_WIDTH;
                float cy = row[1] * INPUT_HEIGHT;
                float w  = row[2] * INPUT_WIDTH;
                float h  = row[3] * INPUT_HEIGHT;

                float x1 = (cx - (w / 2.0f)) * scale_x;
                float y1 = (cy - (h / 2.0f)) * scale_y;
                float x2 = (cx + (w / 2.0f)) * scale_x;
                float y2 = (cy + (h / 2.0f)) * scale_y;

                cv::Scalar color = CLASS_COLORS[class_id];
                std::string label = CLASS_NAMES[class_id] + " " + cv::format("%.2f", confidence);

                cv::rectangle(frame, cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Point(static_cast<int>(x2), static_cast<int>(y2)), color, 3);
                cv::putText(frame, label, cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
            }

            if (frame_count == 0) std::cout << "[DEBUG 9] Opening OpenCV Window...\n";
            cv::imshow("Smart PPE Detection System", frame);
            
            frame_count++;

            if (cv::waitKey(1) == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();
        
    } catch (const Ort::Exception& e) {
        std::cerr << "\n[ONNX RUNTIME FATAL ERROR]: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "\n[STANDARD C++ FATAL ERROR]: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "\n[UNKNOWN FATAL ERROR]: An unexpected error occurred.\n";
    }

    std::cout << "[INFO] Application shutdown complete.\n";
    return 0;
}
import time
import cv2
import torch
from ultralytics import RTDETR

def profile_speed(video_source):
    print("===========================================")
    print("      Hardware Latency & FPS Profiler      ")
    print("===========================================\n")

    # Load your trained PyTorch weights
    model_path = 'C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px-2/weights/best.pt'
    print("[INFO] Loading model into VRAM...")
    model = RTDETR(model_path)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        return

    print("[INFO] Warming up GPU CUDA cores (Processing 10 frames)...")
    for _ in range(10):
        success, frame = cap.read()
        if success:
            # verbose=False stops the terminal from being flooded with print statements
            model(frame, verbose=False) 
            
    print("[INFO] GPU warmed up. Starting strict FPS benchmark...\n")
    
    frame_count = 0
    total_inference_time = 0.0
    
    # Start the overall timer
    pipeline_start_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 1. Start Inference Timer (Measures ONLY the neural network speed)
        inference_start = time.time()
        
        # Run the model with our strict thresholds and NMS rules
        _ = model(frame, stream=True, conf=0.50, iou=0.50, agnostic_nms=True, verbose=False)
        
        # 2. Stop Inference Timer
        inference_end = time.time()
        
        total_inference_time += (inference_end - inference_start)
        frame_count += 1
        
        # Benchmark over 300 frames (roughly 10 seconds of 30fps video) for a stable average
        if frame_count >= 300:
            break
            
    pipeline_end_time = time.time()
    
    # Calculate Metrics
    total_pipeline_time = pipeline_end_time - pipeline_start_time
    
    avg_pipeline_fps = frame_count / total_pipeline_time
    avg_inference_fps = frame_count / total_inference_time
    avg_inference_latency_ms = (total_inference_time / frame_count) * 1000

    print("================ BENCHMARK RESULTS ================")
    print(f"Total Frames Processed:           {frame_count}")
    print(f"Average Inference Latency:        {avg_inference_latency_ms:.2f} ms per frame")
    print(f"Pure Neural Network Speed:        {avg_inference_fps:.2f} FPS")
    print(f"Total Pipeline Speed (with I/O):  {avg_pipeline_fps:.2f} FPS")
    print("===================================================\n")
    print("[NOTE] C++ ONNX Runtime deployment will bypass Python overhead.")
    print("       Expect your final C++ FPS to be up to 1.5x - 2.0x higher.")

    cap.release()

if __name__ == '__main__':
    # Use your simulation video for a consistent benchmark load
    profile_speed("datasets/simulation_video.mp4")
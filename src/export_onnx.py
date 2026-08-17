from ultralytics import RTDETR

def export_model():
    print("===========================================")
    print("     Exporting RT-DETR to ONNX Graph       ")
    print("===========================================\n")

    # Load your best trained PyTorch weights
    model_path = 'C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px-2/weights/best.pt'
    model = RTDETR(model_path)

    # Export to ONNX
    print("[INFO] Converting PyTorch tensors to ONNX format...")
    success = model.export(
        format='onnx',
        imgsz=800,           # Matches your input resolution
        dynamic=False,       # Fixed shape for optimal C++ ONNX Runtime GPU allocation
        simplify=True,       # Simplifies graph nodes for faster C++ parsing
        opset=17             # Standard ONNX opset for RT-DETR compatibility
    )

    print(f"\n[INFO] ONNX Export Completed successfully: {success}")

if __name__ == '__main__':
    export_model()
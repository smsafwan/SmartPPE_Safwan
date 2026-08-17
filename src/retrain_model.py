from ultralytics import RTDETR

def run_finetuning():
    print("===========================================")
    print("   Fine-Tuning RT-DETR on Updated Dataset  ")
    print("===========================================\n")

    # Load your EXISTING trained weights instead of baseline weights
    # This acts as the starting point for fine-tuning
    model_path = 'C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px-2/weights/best.pt'
    model = RTDETR(model_path)

    # Train for 20-30 epochs to adjust bounding box regression
    results = model.train(
        data='datasets/ppe_dataset/data.yaml',
        epochs=25,             # 20-30 epochs is sufficient for box recalibration
        imgsz=800,            # Keep your 800x800 resolution
        batch=2,              # Adjust batch size based on GPU VRAM
        workers=2,
        device=0,
        project='runs/detect/ppe_training_results',
        name='rtdetr_custom_ppe_finetuned',
        exist_ok=True
    )

    print("\n[INFO] Fine-tuning completed successfully!")
    print(f"[INFO] New weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    run_finetuning()
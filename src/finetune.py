from ultralytics import RTDETR

def fast_finetune():
    # 1. Load your CURRENT best weights, not the base model
    model = RTDETR('C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_custom_ppe_finetuned/weights/best.pt')

    # 2. Train for a short duration with optimized settings
    results = model.train(
        data='datasets/ppe_dataset/data.yaml',
        epochs=30,          # Quick fix: only 30 epochs needed to learn the new images
        imgsz=800,
        batch=2,            # Use 8 on the desktop (1070 Ti), drop to 4 if on the laptop (1660 Ti)
        workers=2,          # Keeps the CPU feeding the GPU fast enough
        hsv_h=0.0,          # Locks colors to stop random vest false positives
        patience=10         # Stops early if it finishes learning before 30 epochs
    )

if __name__ == '__main__':
    fast_finetune()
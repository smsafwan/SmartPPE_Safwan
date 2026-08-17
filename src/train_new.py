import os
from ultralytics import RTDETR

def start_training():
    dataset_yaml = 'datasets/ppe_balanced_v2/data.yaml'

    model = RTDETR('rtdetr-l.pt')

    print("==================================================")
    print("      STARTING RT-DETR BALANCED TRAINING RUN     ")
    print("==================================================")

    results = model.train(
        data=dataset_yaml,
        epochs=80,                  # Target ceiling
        patience=20,                # Early stopping
        imgsz=600,                  # Standard 640 resolution (fits 6GB VRAM comfortably)
        batch=4,                    # Stable batch size for 6GB VRAM
        workers=0,                  # FIX: Prevents WinError 1455 virtual memory crash
        hsv_h=0.015,              # Minimal hue shift
        hsv_s=0.5,                  # Saturation
        hsv_v=0.4,                  # Brightness
        mosaic=1.0,                 # Mosaic augmentation
        close_mosaic=15,            # Disable mosaic last 15 epochs
        project='runs/detect/retrain_balanced',
        name='rtdetr_ppe_balanced_v2',
        device=0,
        save=True,
        plots=True
    )

    print("\nTraining completed successfully!")
    print("Best weights saved to: runs/detect/retrain_balanced/rtdetr_ppe_balanced_v2/weights/best.pt")

if __name__ == '__main__':
    start_training()
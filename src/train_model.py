import os
from ultralytics import RTDETR

def main():
    print("========================================")
    print("   Starting RT-DETR Training Pipeline   ")
    print("========================================\n")

    # Load base RT-DETR architecture
    model = RTDETR('rtdetr-l.pt') 

    # Train model optimized for 6GB VRAM & Windows
    results = model.train(
        data='datasets/ppe_dataset/data.yaml', 
        epochs=30,                             
        imgsz=640,                             
        batch=2,                               
        workers=2,
        cache=True,                            # Caches images in RAM for 100% GPU saturation
        device=0,                              
        project='ppe_training_results',
        name='rtdetr_custom_ppe'
    )

    print("\nTraining complete!")

if __name__ == '__main__':
    main()
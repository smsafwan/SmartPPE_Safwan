import os
from ultralytics import RTDETR

def main():
    print("===========================================")
    print("  RT-DETR Front-View & Resolution Pipeline ")
    print("===========================================\n")

    model = RTDETR('rtdetr-l.pt') 

    results = model.train(
        data='datasets/ppe_dataset/data.yaml', 
        epochs=40,                             
        imgsz=800,                             # High resolution to capture front-facing texture
        hsv_h=0.0,  # CRITICAL: Do not randomly change colors during training
        batch=2,                               # Reduced to 1 to fit 800x800 inside 6GB VRAM
        workers=2,                             
        cache=True,                            
        device=0,                              
        
     
        
        project='ppe_training_results',
        name='rtdetr_frontview_800px'
    )

    print("\nTraining complete!")

if __name__ == '__main__':
    main()
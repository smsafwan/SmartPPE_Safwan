from ultralytics import RTDETR

def run_finetuning():
    print("===========================================")
    print("   Retraining RT-DETR on Updated Dataset  ")
    print("===========================================\n")


def run_4class_retraining():
    # Try using your best 5-class weights first
    model = RTDETR('C:/Users/hsmsa/runs/detect/runs/detect/retrain_balanced/rtdetr_ppe_balanced_v2/weights/best.pt')

    model.train(
        data='datasets/retrain_03_09/data.yaml', # <--- Point to the 4-class yaml
        epochs=30,
        imgsz=600,
        batch=4,
        workers=2,
        
        # Robust augmentations for low-light/close-ups
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.7,
        scale=0.8,
        mosaic=1.0,
        
        project='runs/detect/retrain',
        name='rtdetr_ppe_v3'
    )

if __name__ == '__main__':
    run_4class_retraining()
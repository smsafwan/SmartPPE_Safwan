import cv2
import shutil
import torch
from pathlib import Path
from ultralytics import RTDETR

# ==================== CONFIGURATION ====================
# Point exactly to the validation images and labels
SOURCE_VALID_IMAGES = Path('datasets/ppe_balanced_v2/valid/images')
SOURCE_VALID_LABELS = Path('datasets/ppe_balanced_v2/valid/labels')

# Target folder for JUST the CLAHE validation dataset
TARGET_DIR = Path('datasets/ppe_clahe_valid_only')
# =======================================================

def run_clahe_valid_only():
    images_out = TARGET_DIR / 'images'
    labels_out = TARGET_DIR / 'labels'

    # Reset output directory for a clean run
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # ONLY get images from the 'valid/images' folder (No rglob to prevent getting train/test)
    if not SOURCE_VALID_IMAGES.exists():
        print(f"ERROR: Cannot find {SOURCE_VALID_IMAGES}")
        return
        
    image_paths = [
        p for p in SOURCE_VALID_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    print(f"Found {len(image_paths)} images strictly in the 'valid' folder...")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed_count = 0

    for img_path in image_paths:
        target_img_path = images_out / img_path.name
        target_label_path = labels_out / f"{img_path.stem}.txt"

        # 1. Read and apply CLAHE
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(target_img_path), enhanced)

        # 2. Copy matching label file from valid/labels
        source_label = SOURCE_VALID_LABELS / f"{img_path.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, target_label_path)
        else:
            target_label_path.touch() # Empty label for background

        processed_count += 1

    print(f"Successfully processed {processed_count} validation images with CLAHE.")

    # Generate data.yaml pointing directly to the new valid images
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: images
val: images

names:
  0: Bare_Head
  1: NO_Safety_vest
  2: Non_Compliant_Hat
  3: Safety_vest
  4: helmet
"""
    with open(TARGET_DIR / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print("Running evaluation on CLAHE-enhanced validation dataset...")
    
    # Clear CUDA memory before evaluation
    torch.cuda.empty_cache()
    
    model = RTDETR('C:/Users/hsmsa/runs/detect/runs/detect/retrain_balanced/rtdetr_ppe_balanced_v2/weights/best.pt')

    metrics = model.val(
        data=str(TARGET_DIR / 'data.yaml'),
        split='val',
        batch=2,           # Kept low to be safe
        conf=0.55,
        iou=0.60,
        workers=0,
        project='runs/detect/ppe_eval_results',
        name='clahe_valid_only_eval'
    )

    print(f"\nCLAHE mAP@50: {metrics.box.map50:.4f}")
    if len(metrics.box.class_result(1)) >= 3:
        print(f"NO_Safety_vest mAP@50: {metrics.box.class_result(1)[2]:.4f}")

if __name__ == '__main__':
    run_clahe_valid_only()
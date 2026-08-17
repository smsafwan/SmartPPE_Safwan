import shutil
from pathlib import Path

# ==================== CONFIGURATION ====================
# Point exactly to the validation images and labels
SOURCE_VALID_IMAGES = Path('datasets/ppe_balanced_v2/valid/images')
SOURCE_VALID_LABELS = Path('datasets/ppe_balanced_v2/valid/labels')

# Target folder for the consolidated standard dataset
TARGET_DIR = Path('datasets/ppe_valid_consolidated')
# =======================================================

def consolidate_dataset():
    images_out = TARGET_DIR / 'images'
    labels_out = TARGET_DIR / 'labels'

    # Reset output directory for a clean run
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # ONLY get images from the 'valid/images' folder
    if not SOURCE_VALID_IMAGES.exists():
        print(f"ERROR: Cannot find {SOURCE_VALID_IMAGES}")
        return
        
    image_paths = [
        p for p in SOURCE_VALID_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    print(f"Found {len(image_paths)} images strictly in the '{SOURCE_VALID_IMAGES}' folder...")

    processed_count = 0

    for img_path in image_paths:
        target_img_path = images_out / img_path.name
        target_label_path = labels_out / f"{img_path.stem}.txt"

        # 1. Copy the raw image directly
        shutil.copy2(img_path, target_img_path)

        # 2. Copy matching label file from valid/labels
        source_label = SOURCE_VALID_LABELS / f"{img_path.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, target_label_path)
        else:
            target_label_path.touch() # Empty label for background images

        processed_count += 1

    print(f"Successfully copied {processed_count} validation images and labels to '{TARGET_DIR}'.")

    # Generate data.yaml pointing directly to the new consolidated images
    yaml_content = f"""path: {TARGET_DIR.absolute().as_posix()}
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
        
    print(f"Created data.yaml at '{TARGET_DIR / 'data.yaml'}'. Consolidation complete.")

if __name__ == '__main__':
    consolidate_dataset()
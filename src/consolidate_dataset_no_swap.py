import os
import shutil
from pathlib import Path

# ==================== CONFIGURATION ====================
# Input: Path to your current evaluation dataset with subfolders (train, valid, test)
SOURCE_DIR = Path('datasets/ppe_evaluation')

# Output: New directory for the merged dataset
TARGET_DIR = Path('datasets/ppe_evaluation_unified')

# Target Class Names matching your model
CLASS_NAMES = {
    0: 'Bare_Head',
    1: 'NO_Safety_vest',
    2: 'Non_Compliant_Hat',
    3: 'Safety_vest',
    4: 'helmet'
}

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# =======================================================

def consolidate_without_swap():
    images_out = TARGET_DIR / 'images'
    labels_out = TARGET_DIR / 'labels'
    
    # Create output directories (clearing them first if they exist is optional)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Find all image files across all subdirectories
    all_image_paths = [
        p for p in SOURCE_DIR.rglob('*') 
        if p.suffix.lower() in IMAGE_EXTENSIONS and 'unified' not in str(p)
    ]

    print(f"Found {len(all_image_paths)} images across all splits.")

    processed_count = 0
    missing_label_count = 0

    for img_path in all_image_paths:
        # Create a unique filename prefix based on subfolder structure to avoid overwriting
        subfolder_prefix = "_".join(img_path.parent.parts[len(SOURCE_DIR.parts):])
        unique_stem = f"{subfolder_prefix}_{img_path.stem}" if subfolder_prefix else img_path.stem

        target_img_path = images_out / f"{unique_stem}{img_path.suffix.lower()}"
        target_label_path = labels_out / f"{unique_stem}.txt"

        # Copy the image file
        shutil.copy2(img_path, target_img_path)

        # Locate corresponding .txt label file
        possible_label_locations = [
            img_path.parent.parent / 'labels' / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
            img_path.with_suffix('.txt')
        ]

        label_file = next((loc for loc in possible_label_locations if loc.exists()), None)

        if label_file:
            # Copy the label file EXACTLY as it is, with no modifications
            shutil.copy2(label_file, target_label_path)
        else:
            # If no label file exists (background image), create an empty label file
            target_label_path.touch()
            missing_label_count += 1

        processed_count += 1

    print(f"\n--- Consolidation Summary ---")
    print(f"Total Images Copied: {processed_count}")
    print(f"Images without existing label files (created empty .txt): {missing_label_count}")
    print(f"Merged Images Saved To: {images_out.resolve()}")
    print(f"Merged Labels Saved To: {labels_out.resolve()}")

    # Generate unified data.yaml (Includes both train and val keys to pass validation)
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: images
val: images

names:
"""
    for cid, cname in CLASS_NAMES.items():
        yaml_content += f"  {cid}: {cname}\n"

    yaml_path = TARGET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Generated unified data.yaml at: {yaml_path.resolve()}\n")
    print("REMINDER: If re-running evaluation, ensure you delete any old 'labels.cache' file first!")

if __name__ == '__main__':
    consolidate_without_swap()
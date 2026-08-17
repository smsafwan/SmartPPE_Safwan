import os
import glob
from pathlib import Path

def force_swap_labels():
    labels_dir = Path('datasets/ppe_evaluation_unified/labels')
    
    if not labels_dir.exists():
        print(f"Error: Could not find {labels_dir}")
        return

    txt_files = list(labels_dir.glob('*.txt'))
    print(f"Found {len(txt_files)} label files. Swapping Class 1 and Class 2...")

    modified_count = 0

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            
            # Skip empty lines
            if not parts or not parts[0].isdigit():
                new_lines.append(line)
                continue

            cid = int(parts[0])

            # Safely swap 1 and 2
            if cid == 1:
                parts[0] = '2'
                file_changed = True
            elif cid == 2:
                parts[0] = '1'
                file_changed = True

            new_lines.append(" ".join(parts) + "\n")

        # Overwrite the file only if a swap happened
        if file_changed:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            modified_count += 1

    print(f"Successfully fixed {modified_count} label files!")
    print("You are now ready to re-run your evaluation.")

if __name__ == '__main__':
    force_swap_labels()
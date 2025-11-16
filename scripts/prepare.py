# File: scripts/prepare.py
# Description: Unzips raw Roboflow data, splits it into train/val sets,
#              and creates a data.yaml file for YOLO training.
#
# v2: Fixed logic to correctly find labels in .../labels/ directory.

import argparse
import os
import shutil
import zipfile
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def main(args):
    """
    Main function to prepare the dataset.
    """
    
    # --- 1. Define Paths ---
    zip_path = Path(args.zip_file)
    raw_data_dir = Path(args.output_dir) / "raw_unzipped"
    output_dir = Path(args.output_dir)
    
    # Clean up old directories if they exist
    if raw_data_dir.exists():
        shutil.rmtree(raw_data_dir)
    if (output_dir / "train").exists():
        shutil.rmtree(output_dir / "train")
    if (output_dir / "val").exists():
        shutil.rmtree(output_dir / "val")

    # --- 2. Unzip Data ---
    print(f"Unzipping {zip_path} to {raw_data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)
    print("Unzip complete.")

    # --- 3. Find Images and Labels (Corrected Logic) ---
    print("Searching for images and labels...")
    all_images = []
    all_labels = []
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Use rglob to find all 'images' directories (train/images, valid/images, all/images, etc.)
    image_dirs = list(raw_data_dir.rglob('images'))
    
    if not image_dirs:
        print(f"Error: No 'images' directories found in {raw_data_dir}.")
        print("Please check your Roboflow export format.")
        return

    print(f"Found image directories: {[d.relative_to(raw_data_dir) for d in image_dirs]}")

    for img_dir in image_dirs:
        # img_dir is something like .../raw_unzipped/train/images
        # label_dir will be .../raw_unzipped/train/labels
        label_dir = img_dir.parent / 'labels'
        
        if not label_dir.exists():
            print(f"Warning: Found {img_dir} but no corresponding {label_dir}. Skipping.")
            continue
            
        for ext in image_extensions:
            for img_path in img_dir.glob(f'*{ext}'):
                # img_path = .../raw_unzipped/train/images/foo.jpg
                # label_path = .../raw_unzipped/train/labels/foo.txt
                label_path = label_dir / img_path.with_suffix('.txt').name
                
                if label_path.exists():
                    all_images.append(img_path)
                    all_labels.append(label_path)
                else:
                    # This warning means an image is unannotated
                    print(f"Warning: Found image {img_path.name} but no matching label in {label_dir}. Skipping.")
    
    if not all_images:
        print(f"Error: No image/label pairs found in {raw_data_dir}.")
        return
    
    print(f"Found {len(all_images)} total image/label pairs.")

    # --- 4. Split the Data ---
    images_train, images_val, labels_train, labels_val = train_test_split(
        all_images, 
        all_labels, 
        test_size=args.split_ratio, 
        random_state=42 # Use a fixed random state for reproducibility
    )
    
    print(f"Splitting data: {len(images_train)} train, {len(images_val)} validation.")

    # --- 5. Copy Files to Processed Directory ---
    def copy_files(file_list, type, data_type): # e.g., (images_train, "images", "train")
        target_dir = output_dir / data_type / type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in file_list:
            shutil.copy(file_path, target_dir / file_path.name)
            
    print("Copying training files...")
    copy_files(images_train, "images", "train")
    copy_files(labels_train, "labels", "train")
    
    print("Copying validation files...")
    copy_files(images_val, "images", "val")
    copy_files(labels_val, "labels", "val")

    # --- 6. Create data.yaml File ---
    # Read class names from the original Roboflow data.yaml
    source_yaml_path = next(raw_data_dir.rglob('data.yaml'), None)
    
    if source_yaml_path:
        with open(source_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', ['class_0'])
            nc = data_config.get('nc', 1)
        print(f"Loaded {nc} classes: {class_names}")
    else:
        print("Warning: No source data.yaml found. Using default 'class_0'.")
        class_names = ['class_0']
        nc = 1

    # Create the new data.yaml for DVC
    # Paths must be relative for YOLO to work correctly
    yaml_config = {
        'path': str(output_dir.resolve()), # Absolute path to processed data
        'train': 'train/images',           # Relative to 'path'
        'val': 'val/images',               # Relative to 'path'
        'nc': nc,
        'names': class_names
    }
    
    yaml_output_path = output_dir / "data.yaml"
    with open(yaml_output_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully created {yaml_output_path}")
    
    # Clean up the unzipped raw data folder
    shutil.rmtree(raw_data_dir)
    print(f"Cleaned up {raw_data_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from Roboflow zip.")
    
    parser.add_argument("--zip_file", type=str, required=True, 
                        help="Path to the Roboflow .zip file.")
    
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Path to the 'data/processed' directory where outputs will be stored.")
    
    parser.add_argument("--split_ratio", type=float, default=0.2, 
                        help="Ratio of data to use for the validation set (e.g., 0.2 for 20%).")
    
    args = parser.parse_args()
    main(args)

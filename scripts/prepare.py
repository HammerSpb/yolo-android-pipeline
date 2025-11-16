# File: scripts/prepare.py
# Description: Unzips raw Roboflow data, splits it into train/val sets,
#              and creates a data.yaml file for YOLO training.

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

    # --- 3. Find Images and Labels ---
    # This searches recursively for images and matching .txt labels.
    
    all_images = []
    all_labels = []

    print("Searching for images and labels...")
    image_extensions = ['.jpg', '.jpeg', '.png']
    for ext in image_extensions:
        for img_path in raw_data_dir.rglob(f'*{ext}'):
            label_path = img_path.with_suffix('.txt')
            
            if label_path.exists():
                all_images.append(img_path)
                all_labels.append(label_path)
            else:
                print(f"Warning: Found image {img_path} with no matching .txt label. Skipping.")

    if not all_images:
        print(f"Error: No image/label pairs found in {raw_data_dir}.")
        return

    print(f"Found {len(all_images)} image/label pairs.")

    # --- 4. Split the Data ---
    images_train, images_val, labels_train, labels_val = train_test_split(
        all_images, 
        all_labels, 
        test_size=args.split_ratio, 
        random_state=42
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

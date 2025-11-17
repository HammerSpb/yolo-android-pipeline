import os
import shutil
from pathlib import Path

# --- Configuration ---

# This makes the script run from the project's root, not from /scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define all paths relative to the project root
DIRS_TO_REMOVE = [
    "models/fp32/run",
    "models/fp32/best_saved_model",
    "data/raw/dataset",
    "runs"
]

FILES_TO_REMOVE = [
    "models/fp32/best.onnx",
    "calibration_image_sample_data_20x128x128x3_float32.npy"
]

# --- Main Script ---

def main():
    print("Starting project cleanup (Python)...")
    
    # 1. Remove directories
    for dir_path in DIRS_TO_REMOVE:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            shutil.rmtree(full_path, ignore_errors=True)
            print(f"Removed directory: {dir_path}")
        else:
            print(f"Directory not found, skipping: {dir_path}")

    # 2. Remove specific files
    for file_path in FILES_TO_REMOVE:
        full_path = PROJECT_ROOT / file_path
        try:
            os.remove(full_path)
            print(f"Removed file: {file_path}")
        except FileNotFoundError:
            print(f"File not found, skipping: {file_path}")

    # 3. Find and delete all .DS_Store files
    print("Searching for .DS_Store files...")
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file == ".DS_Store":
                ds_store_path = Path(root) / file
                try:
                    os.remove(ds_store_path)
                    print(f"Removed .DS_Store: {ds_store_path.relative_to(PROJECT_ROOT)}")
                except OSError as e:
                    print(f"Error removing .DS_Store: {e}")

    # 4. Create the DVC receipt file
    receipt_path = PROJECT_ROOT / "cleanup_receipt"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.touch()
    
    print("Created DVC receipt.")
    print("Cleanup complete.")

if __name__ == "__main__":
    main()

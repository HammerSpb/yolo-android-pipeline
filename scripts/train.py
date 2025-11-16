# File: scripts/train.py
# Description: Trains a YOLOv8 model using parameters from params.yaml.
#              Saves the best model and final metrics.

import argparse
import json
import os
import shutil
from ultralytics import YOLO
from pathlib import Path

def main(args):
    
    # 1. Load the base model
    print(f"Loading base model: {args.base_model}")
    model = YOLO(args.base_model)

    # 2. Run the training
    print(f"Starting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        project=args.output_dir,
        name='run',  # Hardcode to 'run' for a predictable output path
        exist_ok=True,
        save_json=True
    )
    
    print("Training complete.")

    # 3. Handle Output Files
    training_run_dir = Path(args.output_dir) / 'run'
    weights_dir = training_run_dir / 'weights'
    best_model_path = weights_dir / 'best.pt'
    
    if not best_model_path.exists():
        print(f"ERROR: 'best.pt' was not found in {weights_dir}")
        print("Training may have failed.")
        return

    # --- 4. Move 'best.pt' to the root of the output_dir ---
    # This gives us a clean, predictable path: 'models/fp32/best.pt'
    final_model_path = Path(args.output_dir) / 'best.pt'
    print(f"Moving {best_model_path} to {final_model_path}")
    shutil.move(best_model_path, final_model_path)

    # --- 5. Save Final Metrics to JSON for DVC ---
    # The 'results' object has the final validation metrics
    final_metrics = {
        "mAP50-95": results.box.map,
        "mAP50": results.box.map50,
        "precision": results.box.p[0],
        "recall": results.box.r[0],
        "epochs_trained": args.epochs 
    }
    
    # Ensure the metrics directory exists
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    
    with open(args.output_metrics, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Final training metrics saved to {args.output_metrics}")
    print(json.dumps(final_metrics, indent=4))
    
    # --- 6. (Optional) Clean up the 'run' directory ---
    print(f"Cleaning up {training_run_dir}...")
    shutil.rmtree(training_run_dir)
    print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    
    parser.add_argument("--data", type=str, required=True, 
                        help="Path to the data.yaml file.")
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Base model to start training from (e.g., yolov8n.pt).")
    parser.add_argument("--epochs", type=int, required=True, 
                        help="Number of epochs to train.")
    parser.add_argument("--imgsz", type=int, required=True, 
                        help="Image size for training (e.g., 640).")
    parser.add_argument("--patience", type=int, required=True, 
                        help="Epochs to wait for no improvement.")
    parser.add_argument("--batch", type=int, required=True, 
                        help="Batch size for training.")
    parser.add_argument("--device", type=str, required=True, 
                        help="Device to run on (e.g., 'cpu', '0', 'mps').")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the final 'best.pt' model (e.g., 'models/fp32').")
    parser.add_argument("--output_metrics", type=str, required=True, 
                        help="Path to save the final metrics.json file (e.g., 'metrics/train_stats.json').")
    
    args = parser.parse_args()
    main(args)

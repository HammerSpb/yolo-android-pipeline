# File: scripts/evaluate.py
# Description: Evaluates a model (.pt or .tflite) on multiple criteria:
#             1. mAP (Accuracy)
#             2. Binary Hit-Rate (Real-world accuracy)
#             3. Inference Speed (ms)
#             4. File Size (MB)
#
# v3: Added '--device' argument to run eval on 'mps' (GPU) or 'cpu'.

import argparse
import json
import os
import shutil
import cv2
import time
from ultralytics import YOLO
from pathlib import Path

def run_mAP_evaluation(model, data_yaml_path, imgsz, device): # <--- ADDED device
    """
    Runs the standard YOLO .val() method to get mAP, Precision, Recall.
    """
    print(f"\nRunning mAP evaluation using {data_yaml_path} at {imgsz} on {device}...")
    try:
        # We must specify imgsz for TFLite/ONNX models
        results = model.val(data=data_yaml_path, split='val', imgsz=imgsz, device=device) # <--- ADDED device
        metrics = {
            "mAP50-95": results.box.map,
            "mAP50": results.box.map50,
            "precision": results.box.p[0],
            "recall": results.box.r[0],
        }
        print("mAP evaluation complete.")
    except Exception as e:
        print(f"⚠️ Could not run mAP evaluation: {e}")
        print("    (This can happen with some TFLite models. Reporting 0.)")
        metrics = {"mAP50-95": 0, "mAP50": 0, "precision": 0, "recall": 0}
    return metrics

def run_binary_test(model, test_images_dir, ground_truth_path, output_crops_dir, imgsz, device): # <--- ADDED device
    """
    Runs the 'real-world' binary test (crop vs. no-crop) and
    calculates Hit Rate, False Alarm Rate, and Inference Speed.
    """
    print(f"\nRunning binary 'real-world' test on {device}...")
    
    # --- 1. Load Ground Truth ---
    try:
        with open(ground_truth_path, 'r') as f:
            truth_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Could not find ground truth file: {ground_truth_path}. Skipping binary test.")
        return {}

    # --- 2. Setup Directories and Counters ---
    images_dir = Path(test_images_dir)
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"⚠️ No images found in {test_images_dir}. Skipping binary test.")
        return {}
        
    if os.path.exists(output_crops_dir):
        shutil.rmtree(output_crops_dir)
    os.makedirs(output_crops_dir, exist_ok=True)

    tp, fp, tn, fn = 0, 0, 0, 0
    total_inference_time_ms = 0
    image_count = 0

    # --- 3. Warm-up Run (for accurate speed testing) ---
    print(f"Running model warm-up for speed test at {imgsz} on {device}...")
    try:
        warmup_img_path = str(image_files[0])
        warmup_img = cv2.imread(warmup_img_path)
        if warmup_img is not None:
            model(warmup_img, verbose=False, imgsz=imgsz, device=device) # <--- ADDED device
        else:
            print(f"⚠️ Could not read warm-up image: {warmup_img_path}")
    except Exception as e:
        print(f"⚠️ Model warm-up failed: {e}")

    # --- 4. Process Each Image ---
    print(f"Found {len(image_files)} test images in {images_dir}")
    for idx, image_path in enumerate(sorted(image_files)):
        if image_path.name not in truth_data:
            print(f"[{idx+1}/{len(image_files)}] ⚠️ Skipping {image_path.name}: Not found in ground_truth.json")
            continue

        ground_truth = truth_data[image_path.name] # "hit" or "miss"
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[{idx+1}/{len(image_files)}] ⚠️ Could not read {image_path.name}")
            continue

        # --- Run inference and time it ---
        start_time = time.perf_counter()
        results = model(img, verbose=False, imgsz=imgsz, device=device) # <--- ADDED device
        end_time = time.perf_counter()
        
        total_inference_time_ms += (end_time - start_time) * 1000  # Convert to ms
        image_count += 1
        model_detected = bool(results and len(results[0].boxes) > 0)
        # ----------------------------------

        # --- 5. Compare Model vs. Ground Truth ---
        if ground_truth == "hit" and model_detected:
            tp += 1
        elif ground_truth == "miss" and model_detected:
            fp += 1
        elif ground_truth == "miss" and not model_detected:
            tn += 1
        elif ground_truth == "hit" and not model_detected:
            fn += 1

        # --- 6. Save Crops (if any) ---
        if model_detected:
            for det_id, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                crop = img[y1:y2, x1:x2]
                crop_name = f"{Path(image_path).stem}_win{det_id+1}_{conf:.2f}.jpg"
                crop_path = os.path.join(output_crops_dir, crop_name)
                cv2.imwrite(crop_path, crop)

    # --- 7. Calculate Final Binary Metrics ---
    total_hits = tp + fn
    total_misses = tn + fp
    
    hit_rate = (tp / total_hits) if total_hits > 0 else 0
    false_alarm_rate = (fp / total_misses) if total_misses > 0 else 0
    avg_inference_ms = (total_inference_time_ms / image_count) if image_count > 0 else 0
    
    binary_metrics = {
        "Hit_Rate": hit_rate,
        "False_Alarm_Rate": false_alarm_rate,
        "avg_inference_ms": avg_inference_ms,
        "True_Positives": tp,
        "False_Positives": fp,
        "True_Negatives": tn,
        "False_Negatives": fn
    }
    
    print(f"Binary test complete. Crops saved to {output_crops_dir}")
    return binary_metrics

def main(args):
    # --- 1. Get Model Size ---
    try:
        model_size_mb = os.path.getsize(args.weights) / (1024 * 1024)
    except FileNotFoundError:
        print(f"⚠️ Could not find model file at {args.weights} to get size.")
        model_size_mb = 0
    
    # 2. Load the model
    print(f"Loading model from {args.weights} (Size: {model_size_mb:.2f} MB)")
    model = YOLO(args.weights)

    # 3. Run mAP evaluation
    map_metrics = run_mAP_evaluation(model, args.data, args.imgsz, args.device) # <--- ADDED args.device

    # 4. Run binary "real-world" test
    binary_metrics = run_binary_test(
        model,
        args.test_images_dir,
        args.test_ground_truth,
        args.output_crops_dir,
        args.imgsz,
        args.device  # <--- ADDED args.device
    )

    # 5. Combine all metrics
    all_metrics = {
        "model_size_mb": model_size_mb,
        **map_metrics,
        **binary_metrics
    }
    
    # 6. Save combined metrics to the output file
    os.makedirs(os.path.dirname(args.output_metrics_file), exist_ok=True)
    with open(args.output_metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n✅ All evaluations complete. Combined metrics saved to {args.Soutput_metrics_file}")
    print(json.dumps(all_metrics, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model (mAP, binary, speed, size) and save metrics.")
    
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--data", type=str, required=True, help="Path to the data.yaml file.")
    
    parser.add_argument("--test_images_dir", type=str, required=True, help="Path to the 'real-world' test images.")
    parser.add_argument("--test_ground_truth", type=str, required=True, help="Path to the 'ground_truth.json' file.")
    parser.add_argument("--output_crops_dir", type=str, required=True, help="Path to save cropped image results.")
    
    parser.add_argument("--output_metrics_file", type=str, required=True, help="Path to save the combined output metrics.json.")
    
    parser.add_argument("--imgsz", type=int, required=True, help="Image size for evaluation (e.g., 640).")
    
    parser.add_argument("--device", type=str, required=True, help="Device to run on (e.g., 'cpu', 'mps').") # <--- ADDED
    
    args = parser.parse_args()
    main(args)

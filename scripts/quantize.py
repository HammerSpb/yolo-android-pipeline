# File: scripts/quantize.py
# Description: Loads an FP32 .pt model and exports it to a quantized
#              TFLite format (FP16 or INT8).

import argparse
import os
import shutil
from ultralytics import YOLO

def main(args):
    """
    Loads an FP32 .pt model and exports it to TFLite (FP16 or INT8).
    """
    
    # 1. Load the FP32 model
    print(f"Loading base model from {args.weights}...")
    model = YOLO(args.weights)

    # 2. Set up export parameters
    export_params = {
        'format': args.format
    }

    # 3. Configure for FP16 or INT8
    if args.mode == 'fp16':
        export_params['half'] = True
        print(f"Configured for FP16 export ({args.format})...")
    elif args.mode == 'int8':
        export_params['int8'] = True
        if not args.data:
            raise ValueError("INT8 quantization requires a --data yaml file for calibration.")
        export_params['data'] = args.data
        print(f"Configured for INT8 export ({args.format}) with calibration data...")
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'fp16' or 'int8'.")

    # 4. Perform the export
    try:
        exported_model_path = model.export(**export_params)
        print(f"Model exported by Ultralytics to: {exported_model_path}")
    except Exception as e:
        print(f"An error occurred during export: {e}")
        return

    # 5. Move the exported model to the location DVC expects
    # This gives us a clean, predictable file path for our DVC 'outs'.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    shutil.move(exported_model_path, args.output_file)
    
    print(f"Moved quantized model to final path: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a YOLO .pt model to TFLite.")
    
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to the input FP32 .pt model.")
    
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Desired final path for the quantized model (e.g., 'models/fp16/best.tflite').")
    
    parser.add_argument("--mode", type=str, required=True, choices=['fp16', 'int8'], 
                        help="Quantization mode: 'fp16' or 'int8'.")
    
    parser.add_argument("--format", type=str, default="tflite", 
                        help="Export format (should be 'tflite' for this pipeline).")
    
    parser.add_argument("--data", type=str, 
                        help="Path to data.yaml. REQUIRED for INT8 calibration.")
    
    args = parser.parse_args()
    main(args)

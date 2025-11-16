
YOLOv8 MLOps Pipeline for Android (TFLite)

This repository contains a complete, reproducible MLOps pipeline for training, evaluating, and quantizing YOLOv8 models for Android. It takes raw annotated data from Roboflow, trains a model, and automatically converts, evaluates, and compares FP32, FP16, and INT8 TFLite models to find the best one for production.

This pipeline is built to be modular, track experiments, and bridge the gap between training on a powerful GPU (like in Google Colab) and evaluating on a local machine (like a Mac).

Core Technologies

Git: For tracking all code and configuration.

DVC (Data Version Control): For versioning large files (datasets, models) and managing the pipeline.

Python: For all custom scripts.

Ultralytics YOLOv8: For training and exporting models.

Conda: For managing the Python environment.

Project Structure

yolo-android-pipeline/
├── .dvc/                   # DVC's internal files
├── data/
│   ├── raw/                # Raw Roboflow .zip (DVC-tracked)
│   │   └── dataset.zip
│   ├── processed/          # Split train/val data (DVC-tracked)
│   │   ├── train/
│   │   ├── val/            # This set is used for INT8 calibration
│   │   └── data.yaml
│   ├── real_test/          # Binary (hit/miss) test set (DVC-tracked)
│   │   ├── images/
│   │   └── ground_truth.json
│   └── crops/              # Cropped images from evaluation (DVC-tracked)
├── models/                 # Output models (DVC-tracked)
│   ├── fp32/ (best.pt)
│   ├── fp16/ (best.tflite)
│   └── int8/ (best.tflite)
├── metrics/                # Output metrics (DVC-tracked)
│   ├── eval_fp32.json
│   ├── eval_fp16.json
│   └── eval_int8.json
├── scripts/                # All pipeline scripts (Git-tracked)
│   ├── prepare.py
│   ├── train.py
│   ├── quantize.py
│   └── evaluate.py
├── .gitignore              # Ignores data/models from Git
├── dvc.yaml                # The DVC pipeline "brain"
├── params.yaml             # Experiment parameters
└── requirements.txt        # Python libraries

Scripts Overview

scripts/prepare.py: Takes the raw data/raw/dataset.zip, unzips it, splits it into train and val sets, and creates the data/processed/data.yaml file for YOLO.

scripts/train.py: Trains the FP32 .pt model using parameters from params.yaml and the data.yaml.

scripts/quantize.py: Takes the trained .pt model and exports it to TFLite (FP16 and INT8), using the val set for INT8 calibration.

scripts/evaluate.py: The all-in-one evaluation script. It runs on a model (TFLite or .pt) and measures:

Accuracy (mAP): Standard precision/recall metrics.

Accuracy (Binary): A custom "real-world" test (Hit Rate, False Alarm Rate) based on data/real_test/.

Speed: Average inference time (ms) on your local machine.

Size: Model file size (MB).

How to Use This Pipeline

1. Setup & Installation (First Time)

Clone the Repo:

git clone <your-repo-url>
cd yolo-android-pipeline

Create & Activate Environment:

conda create -n yolo_pipeline python=3.10
conda activate yolo_pipeline
pip install -r requirements.txt

Connect DVC Remote:

You only need to do this once. This example uses Google Drive.

dvc remote modify --local gdrive auth true
dvc remote modify --local gdrive use_pydrive true

# This will prompt you to authenticate in your browser

Pull Data & Models:

This downloads the current production-ready data and models.

dvc pull

2. Running the Full Pipeline (Local Mac)

This command will run all stages (prepare, train, quantize, evaluate) if any code or data has changed. DVC is smart and will skip any stage that is already up-to-date.

dvc repro

3. Running Experiments (The Main Workflow)

This is the most powerful part. Use dvc exp run to test new ideas by overriding parameters in params.yaml.

Example: Test a yolov8s model (vs. yolov8n)

dvc exp run --name "test-yolov8s" -S "train.base_model=yolov8s.pt"

Example: Test with more epochs and a different batch size

dvc exp run --name "long-run-16batch" -S "train.epochs=100" -S "train.batch=16"

Compare All Your Experiments
This command shows a dashboard comparing the parameters and all your TFLite model metrics (Hit Rate, speed, size).

dvc exp show

Example Output:

| Experiment | params.yaml:train.base_model | metrics/eval_int8.json:Hit_Rate | metrics/eval_int8.json:avg_inference_ms | metrics/eval_int8.json:model_size_mb |
| :--- | :--- | :--- | :--- | :--- |
| baseline | yolov8n.pt | 0.97 | 14.5 | 1.9 |
| test-yolov8s| yolov8s.pt | 0.99 | 22.1 | 3.4 |

4. Partial Run (Google Colab + Local Mac)

Use this when you want to train on a powerful GPU but evaluate on your local machine.

Step A: In Google Colab (Training)

# 1. Mount Drive & Clone Repo

from google.colab import drive
drive.mount('/content/drive')
!git clone <your-repo-url>
%cd yolo-android-pipeline

# 2. Install Libs

!pip install -r requirements.txt

# 3. Pull ONLY the training data

!dvc pull data/processed.dvc

# 4. Run ONLY the 'train' stage on the GPU

!dvc repro train

# 5. Push the NEW trained model back to DVC remote

!dvc push models/fp32/best.pt.dvc

Step B: On Your Local Mac (Quantize & Evaluate)

# 1. Pull the new model trained by Colab

dvc pull

# 2. Run the rest of the pipeline

dvc repro

DVC will see that models/fp32/best.pt is new and will automatically run quantize and evaluate. It will skip prepare_data and train, saving you hours.

5. Deploying to Android

Once dvc exp show reveals your best model, promote it.

Apply & Tag: (e.g., your best experiment was test-yolov8s)

dvc exp apply test-yolov8s
git commit -m "feat: Promote yolov8s model to production (v1.0)"
git tag -a "android-v1.0" -m "Production model v1.0, 99% hit rate"
git push origin main --tags

Get the Model for Your App:

In your Android project, you can pull the file directly by its tag:

# 1. Checkout the tag

git checkout android-v1.0

# 2. Pull the one TFLite file you need

dvc pull models/int8/best.tflite.dvc

# 3. The 'models/int8/best.tflite' file is now ready

# Copy it into your Android Studio assets folder

Roboflow Dataset Configuration
For this pipeline to work correctly, the dataset downloaded from Roboflow must be configured in a specific way. The pipeline is designed to be the "source of truth" for data splitting and augmentation, not Roboflow.

When creating a new version of your dataset in Roboflow, use these exact settings before generating your .zip file:

1. Train/Test Split

Set everything to 100% "Training". Do not create validation or test splits. Our scripts/prepare.py script handles this reproducibly.

Training Set: 100%

Validation Set: 0%

Testing Set: 0%

2. Preprocessing

Use only Auto-Orient. This fixes image rotation issues from camera metadata.

Do NOT Resize. Do not use any Resize step (e.g., "Fit 640x640"). Our train.py script handles resizing and letterboxing in memory, which is more flexible and effective.

Steps to Apply:

Auto-Orient

3. Augmentation

Add 0 Augmentation Steps.

Do NOT add any augmentations (like Flip, Saturation, Brightness, Mosaic, etc.). The YOLOv8 train() function already applies a superior set of augmentations in memory during training. Adding them here will bloat your dataset and interfere with the training process.

Steps to Apply: (None)

4. Export Format

After generating the version, click "Export" and choose the "YOLO v8" format. This provides the dataset.zip file (containing images/, labels/, and data.yaml) that our pipeline is built to understand.:wq

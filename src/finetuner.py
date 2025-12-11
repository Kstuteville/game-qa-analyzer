"""
finetuner.py

Utility functions for YOLOv8 fine-tuning:
- extract_zip(zip_path, output_dir)
- create_dataset_yaml(dataset_root)
- train_yolov8(base_model, data_yaml, epochs, imgsz, save_path)
"""

import zipfile
import os
from pathlib import Path
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. Extract dataset ZIP
# ---------------------------------------------------------
def extract_zip(zip_path: str, output_dir: str) -> str:
    """
    Extracts a YOLO dataset .zip file into output_dir.
    Returns the path to the extracted dataset folder.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)

    # Find the dataset root (first folder)
    for p in output_dir.iterdir():
        if p.is_dir():
            return str(p)

    return str(output_dir)


# ---------------------------------------------------------
# 2. Create dataset YAML file
# ---------------------------------------------------------
def create_dataset_yaml(dataset_root: str) -> str:
    """
    Generates a YOLO dataset YAML file pointing to train/val sets.
    Returns path to the YAML file.
    """

    dataset_root = Path(dataset_root)

    yaml_dict = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {}
    }

    # Infer class names from labels directories
    label_dir = dataset_root / "labels" / "train"
    classes = set()

    if label_dir.exists():
        for f in label_dir.glob("*.txt"):
            with open(f, "r") as file:
                contents = file.read().strip().splitlines()
                for line in contents:
                    if line:
                        class_id = int(line.split()[0])
                        classes.add(class_id)

    yaml_dict["names"] = {i: f"class_{i}" for i in sorted(classes)}

    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    return str(yaml_path)


# ---------------------------------------------------------
# 3. Train YOLOv8 model
# ---------------------------------------------------------
def train_yolov8(base_model: str, data_yaml: str, epochs: int, imgsz: int, save_path: str):
    """
    Fine-tunes a YOLOv8 model.
    Saves the trained model to save_path.
    """

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        pretrained=True
    )

    # Save best model
    best = model.ckpt_path
    if best and Path(best).exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        os.replace(best, save_path)

    return results

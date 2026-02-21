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

def extract_zip(zip_path: str, output_dir: str) -> str:
   """
   Extracts a YOLO dataset .zip file into output_dir.
   Returns the path to the extracted dataset folder.
   """
   zip_path = Path(zip_path)
   output_dir = Path(output_dir)


   with zipfile.ZipFile(zip_path, "r") as z:
       z.extractall(output_dir)
   # Return the FIRST folder extracted
   for p in output_dir.iterdir():
       if p.is_dir():
           return str(p)


   return str(output_dir)

def create_dataset_yaml(dataset_root: str) -> str:
   """
   Generate a YOLO data.yaml compatible with your dataset structure.
   dataset_root will be something like: data/training/train
   We FIX this by using dataset_root.parent → data/training
   """

   dataset_root = Path(dataset_root)
   yaml_output_dir = dataset_root.parent

   label_dirs = [
       yaml_output_dir / "train" / "labels",
       yaml_output_dir / "valid" / "labels",
       yaml_output_dir / "test" / "labels",
   ]
   classes = set()
   for label_dir in label_dirs:
       if label_dir.exists():
           for f in label_dir.glob("*.txt"):
               with open(f, "r") as file:
                   for line in file.read().strip().splitlines():
                       if line:
                           class_id = int(line.split()[0])
                           classes.add(class_id)

   if not classes:
       print("WARNING: No label classes detected. Defaulting to 1-class dataset ('enemy').")
       classes = {0}
   names_dict = {i: f"class_{i}" for i in sorted(classes)}

   yaml_dict = {
       "path": str(yaml_output_dir),
       "train": "train/images",
       "val": "valid/images",
       "test": "test/images",
       "nc": len(names_dict),
       "names": names_dict
   }
   yaml_path = yaml_output_dir / "data.yaml"
   with open(yaml_path, "w") as f:
       yaml.dump(yaml_dict, f)


   print(f"Dataset YAML created at: {yaml_path}")
   print(f"Classes detected: {names_dict}")


   return str(yaml_path)

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
   best = model.ckpt_path
   if best and Path(best).exists():
       Path(save_path).parent.mkdir(parents=True, exist_ok=True)
       os.replace(best, save_path)
   return results



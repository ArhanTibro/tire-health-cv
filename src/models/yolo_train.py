
"""
Fine-tune YOLO11-cls (Ultralytics) on the 5-class tire dataset.

Reusable training/eval helpers. The notebook (03_yolo_transfer.ipynb) calls
these functions - it does NOT call ultralytics directly, so the workflow stays
config-driven and consistent with the other models.
"""

import os
import shutil
import torch
from ultralytics import YOLO


def build_yolo_model(model_name="yolo11n-cls.pt"):
    """Load a pretrained YOLO11 classification model."""
    return YOLO(model_name)


def train_yolo(
    data_dir,
    model_name="yolo11n-cls.pt",
    epochs=50,
    imgsz=224,
    batch=16,
    lr0=0.001,
    freeze=None,
    project="../results/yolo11n",
    name="train",
    seed=42,
):
    """
    Fine-tune YOLO11-cls.

    data_dir: path to a directory containing train/ and val/ subfolders,
              each with one subfolder per class. This is exactly the
              structure at data/processed/.
    freeze:   int N = freeze first N layers (transfer learning).
              None = train all layers.
    """
    model = build_yolo_model(model_name)

    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        freeze=freeze,
        project=project,
        name=name,
        seed=seed,
        exist_ok=True,
        verbose=True,
    )

    return model, results


def get_yolo_predictions(model, image_paths, class_names):
    """
    Run inference on a list of image paths and return (true_labels, predicted_labels)
    in the same format src/utils/metrics.py expects, so YOLO results are directly
    comparable to CNN/ResNet results.
    """
    true_labels = []
    predicted_labels = []

    for path in image_paths:
        true_class = os.path.basename(os.path.dirname(path))
        true_labels.append(class_names.index(true_class))

        result = model.predict(path, verbose=False)[0]
        pred_index = result.probs.top1
        predicted_labels.append(pred_index)

    return true_labels, predicted_labels


def collect_image_paths(split_dir, class_names):
    """Gather all image paths + true labels from a split folder."""
    image_paths = []
    for class_name in class_names:
        class_folder = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_folder, fname))
    return image_paths


def copy_best_weights(run_dir, dest_path):
    """Copy best.pt from the Ultralytics run folder to results/yolo11n/model.pt."""
    src = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src, dest_path)
        print(f"Copied best weights: {src} -> {dest_path}")
        return True
    print(f"No best.pt found at {src}")
    return False
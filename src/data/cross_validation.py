import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class ListDataset(Dataset):
    # simple dataset that loads images from a plain list of file paths and labels
    # (regular ImageFolder needs a fixed folder structure - this doesn't)
    def __init__(self, image_paths, labels, class_names, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        label = self.class_names.index(self.labels[index])
        return image, label


def get_real_images(processed_folder):
    # gathers all REAL images from train + val (skips files starting with "aug_")
    # test folder is intentionally left out - k-fold never touches it
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(os.path.join(processed_folder, "train")))

    for split_name in ["train", "val"]:
        split_folder = os.path.join(processed_folder, split_name)
        for class_name in class_names:
            class_folder = os.path.join(split_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            for file_name in os.listdir(class_folder):
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                if file_name.startswith("aug_"):
                    continue
                image_paths.append(os.path.join(class_folder, file_name))
                labels.append(class_name)

    return image_paths, labels, class_names


def make_folds(image_paths, labels, k=5, seed=42):
    # returns a list of (train_indices, val_indices), one pair per fold
    # stratified means each fold keeps roughly the same class ratio as the whole set
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(skf.split(image_paths, labels))
    return folds
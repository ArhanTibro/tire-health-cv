"""
Rename a folder of newly collected images to the project's naming convention:

    <class>_<collector-initials>_<number>.<ext>

Usage:
    python rename_images.py --folder data/raw/healthy --class_name Healthy --initials AR
"""
import argparse
import os


def rename_images(folder_path: str, class_name: str, collector_initials: str) -> None:
    valid_ext = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_ext
    )
    for i, fname in enumerate(files, 1):
        ext = os.path.splitext(fname)[1].lower()
        new_name = f"{class_name}_{collector_initials}_{i:03d}{ext}"
        os.rename(os.path.join(folder_path, fname), os.path.join(folder_path, new_name))
    print(f"Renamed {len(files)} files in {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to the class folder")
    parser.add_argument("--class_name", required=True, help="e.g. Healthy, Low_Tread")
    parser.add_argument("--initials", required=True, help="Collector's initials, e.g. AR")
    args = parser.parse_args()
    rename_images(args.folder, args.class_name, args.initials)

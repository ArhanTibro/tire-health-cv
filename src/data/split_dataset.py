import os
import shutil
import random

def check_dataset(folder_path):
    class_names = os.listdir(folder_path)
    counts = {}
    total = 0

    for class_name in class_names:
        if class_name == "expired":
            continue
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[class_name] = len(images)
        total += len(images)

    print("Dataset check:")
    for class_name, count in counts.items():
        warning = ""
        if count < 20:
            warning = "  <- low, may need more images"
        print(class_name, ":", count, "images", warning)
    print("Total:", total, "images")

    return counts


def split_data(input_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    class_names = os.listdir(input_folder)

    for class_name in class_names:
        if class_name == "expired":
            continue
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        copy_images(train_images, class_path, output_folder, "train", class_name)
        copy_images(val_images, class_path, output_folder, "val", class_name)
        copy_images(test_images, class_path, output_folder, "test", class_name)

        print(class_name, "- train:", len(train_images), "val:", len(val_images), "test:", len(test_images))


def copy_images(image_list, source_folder, output_folder, split_name, class_name):
    target_folder = os.path.join(output_folder, split_name, class_name)
    os.makedirs(target_folder, exist_ok=True)

    for image in image_list:
        src = os.path.join(source_folder, image)
        dst = os.path.join(target_folder, image)
        shutil.copy(src, dst)


if __name__ == "__main__":
    input_folder = "../data/softcom_dataset"
    output_folder = "../data/processed"

    check_dataset(input_folder)
    print()
    split_data(input_folder, output_folder)
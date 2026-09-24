import os
import random
from PIL import Image, ImageEnhance

def augment_image(image):
    choice = random.randint(1, 3)

    if choice == 1:
        angle = random.randint(-15, 15)
        image = image.rotate(angle)

    elif choice == 2:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    else:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

    return image


def balance_train_folder(train_folder):
    random.seed(42)  # ADDED - same augmented images every time this runs

    class_names = os.listdir(train_folder)
    counts = {}

    for class_name in class_names:
        class_path = os.path.join(train_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        images = os.listdir(class_path)
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[class_name] = len(images)

    target = max(counts.values())
    print("Target count per class:", target)

    for class_name, count in counts.items():
        class_path = os.path.join(train_folder, class_name)
        images = os.listdir(class_path)
        images = [f for f in images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        needed = target - count
        if needed <= 0:
            print(class_name, "- already at target, skipping")
            continue

        print(class_name, "- adding", needed, "augmented images")

        count_added = 0
        while count_added < needed:
            original_name = random.choice(images)
            original_path = os.path.join(class_path, original_name)

            image = Image.open(original_path).convert("RGB")
            augmented_image = augment_image(image)

            new_name = "aug_" + str(count_added) + "_" + original_name
            new_path = os.path.join(class_path, new_name)
            augmented_image.save(new_path)

            count_added += 1


if __name__ == "__main__":
    balance_train_folder("../data/processed/train")
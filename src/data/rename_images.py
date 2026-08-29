import os

def rename_images(folder_path, class_name):
    files = os.listdir(folder_path)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()

    count = 1
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        new_name = class_name + "_" + str(count).zfill(3) + ext
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        count += 1

    print("Renamed", count - 1, "files in", folder_path)


if __name__ == "__main__":
    folder = input("Folder path: ")
    class_name = input("Class name (e.g. Healthy): ")
    rename_images(folder, class_name)
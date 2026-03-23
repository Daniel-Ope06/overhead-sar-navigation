import os
import random
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "synthetic_dataset")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test ratio is automatically the remainder


def auto_flatten(dirs):
    """Pulls all files out of train/val/test subfolders
    back to the root folder.
    """
    print("[*] Flattening directories for a fresh shuffle...")
    for root_dir in dirs.values():
        if not os.path.exists(root_dir):
            continue
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(root_dir, split)
            if os.path.exists(split_dir):
                for filename in os.listdir(split_dir):
                    file_path = os.path.join(split_dir, filename)
                    if os.path.isfile(file_path):
                        # Move file back up one level to the root directory
                        shutil.move(file_path, os.path.join(
                            root_dir, filename))


def split_data():
    dirs = {
        "images": os.path.join(DATASET_DIR, "images"),
        "labels": os.path.join(DATASET_DIR, "labels"),
        "masks": os.path.join(DATASET_DIR, "masks")
    }

    # Reset the folders & files
    auto_flatten(dirs)
    # return  # Keep to only flatten

    # Create the Train/Val/Test subdirectories
    for dir_path in dirs.values():
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dir_path, split), exist_ok=True)

    # Grab all image files and shuffle them randomly
    all_images = [f for f in os.listdir(dirs["images"]) if f.endswith('.png')]
    random.shuffle(all_images)
    total_images = len(all_images)

    # Calculate split indices
    train_idx = int(total_images * TRAIN_RATIO)
    val_idx = train_idx + int(total_images * VAL_RATIO)

    print(
        f"[*] Splitting {total_images} files ->" +
        f"Train: {train_idx} | " +
        f"Val: {val_idx - train_idx} | " +
        f"Test: {total_images - val_idx}")

    # Move the files to their new designated folders
    for i, img_name in enumerate(all_images):
        if i < train_idx:
            split = 'train'
        elif i < val_idx:
            split = 'val'
        else:
            split = 'test'

        # Move the Image
        shutil.move(os.path.join(dirs["images"], img_name),
                    os.path.join(dirs["images"], split, img_name))

        # Move the YOLO Bounding Box Label
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(dirs["labels"], label_name)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(
                dirs["labels"], split, label_name))

        # Move the U-Net Segmentation Mask
        mask_path = os.path.join(dirs["masks"], img_name)
        if os.path.exists(mask_path):
            shutil.move(mask_path, os.path.join(
                dirs["masks"], split, img_name))

    print("[+] Dataset successfully organized into Train/Val/Test splits.")


if __name__ == "__main__":
    split_data()

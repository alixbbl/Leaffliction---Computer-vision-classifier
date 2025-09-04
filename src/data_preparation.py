import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split  # Split données
from Augmentation import folder_augmentation
from config import OUTPUT_DIR

# Dans ce script on doit inclure :

# 🧮 Split	split_dataset(...)
# Sépare les images en train, val, test de manière stratifiée

# 📁 Chargement	load_dataset()
# Charge les images depuis un dossier train

# 🧪 Préprocessing	transform = transforms.Compose(...)
# Applique ToTensor, resize, normalisation, et potentiellement des effets

# 💾 Sauvegarde	save_dataset(...)
# Sauvegarde chaque split dans un sous-dossier avec les classes

# 🧰 Interface CLI	argparse
# Le script est utilisable depuis la ligne de commande


def get_X_y(folder_src: Path) -> Tuple[List[str], List[str]]:
    """
        Colects images paths and labels in order to create the training
        dataset.
        input: source folder path.
        output:
    """
    image_paths = []
    labels = []

    for class_dir in os.listdir(folder_src):
        class_path = os.path.join(folder_src, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_file))
                    labels.append(class_dir)

    return image_paths, labels


def save_folders(labels: List[str], train_paths: List[str],
                 val_paths: List[str],
                 train_labels: List[str], val_labels: List[str]) -> None:
    """
        Save the images into new folders, according to the split previously
        performed.
        input:
        output: None
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for label in set(labels):
            os.makedirs(os.path.join(split_dir, label), exist_ok=True)

    for path, label in zip(train_paths, train_labels):
        dst = os.path.join(OUTPUT_DIR, 'train', label, os.path.basename(path))
        shutil.copy2(path, dst)
    for path, label in zip(val_paths, val_labels):
        dst = os.path.join(OUTPUT_DIR, 'val', label, os.path.basename(path))
        shutil.copy2(path, dst)

    print("Physical split achieved!")


def split_data(folder_src: Path, val_ratio=0.2, random_state=42) -> None:
    """
        Use the train_test_split Scikit Learn function to compute a physical
        split.
        input: source folder path and scikit learn function cnstants.
        output: None
    """
    image_paths, labels = get_X_y(folder_src)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=labels
    )
    save_folders(labels, train_paths, val_paths, train_labels, val_labels)


# =================================== MAIN ===================================

def main(parsed_args):

    folder_src = Path(parsed_args.folder_src)
    if not folder_src.is_dir():
        print("Not a directory !")
        return
    try:
        augmented_folder = Path(OUTPUT_DIR) / "augmented_directory"
        augmented_root = augmented_folder / folder_src.name
        shutil.copytree(folder_src, augmented_root, dirs_exist_ok=True)

        folder_augmentation(str(augmented_root))

        print("Splitting augmented data into train/val...")
        split_data(augmented_root)
        # train_folder = Path(OUTPUT_DIR) / "train"
        # folder_transformation(str(train_folder), train_folder)
        # A VOIR SI ON LE FAIT

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('folder_src',
                        type=str,
                        help="Source folder path.")
    parsed_args = parser.parse_args()
    main(parsed_args)

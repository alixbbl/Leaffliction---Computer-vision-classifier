import argparse
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os

augmentations = {

    "Flip": transforms.RandomHorizontalFlip(p=1.0),
    "Rotate": transforms.RandomRotation(degrees=45),
    "Crop": transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0)),
    "Shear": transforms.RandomAffine(degrees=0, shear=15),
    "Skew": transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
    "Distortion": transforms.ColorJitter(brightness=0.5, contrast=0.5)
}

# ============================== File Augmentation ================================

def file_augmentation(image_path: str) -> None:
    
    directory_name = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    image = Image.open(image_path)
    for augmentation_name, augmentation_effect in augmentations.items():
        imaged_augmented = augmentation_effect(image)

        new_image_name = f"{filename}_{augmentation_name}.JPG"
        output_path = os.path.join(directory_name, new_image_name)
        imaged_augmented.save(output_path)
        plt.imshow(imaged_augmented)
        plt.title(augmentation_name)
        plt.show()

# ============================= Folder Augmentation ===============================

def folder_augmentation(folder_path: str) -> None:
    pass


# ===================================== MAIN ======================================

def main(parsed_args):
    
    try:
        if parsed_args.image_path:
            if not parsed_args.image_path.lower().endswith((".jpg", ".jpeg")):
                print("Not a valid image format!")
                return
            file_augmentation(parsed_args.image_path)
            
        elif parsed_args.directory_path:
            if not Path(parsed_args.directory_path).is_dir():
                print("Not a directory!")
                return
            folder_augmentation(parsed_args.directory_path)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str,
                        default=None,
                        help="Selected image for a 6 ways data-augmentation.")
    parser.add_argument('--directory_path',
                        type=str,
                        default=None,
                        help="Option to be used in the training part.")
    parsed_args = parser.parse_args()
    main(parsed_args)
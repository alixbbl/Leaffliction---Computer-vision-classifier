import argparse, os, random
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from Distribution import stats_from_dir
random.seed(42)

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
    """
        Operate a 6 ways data-augmentation for a single image.
        input: path of the image
        output: None
        The 6 new images are located in the original image folder.
    """
    directory_name = os.path.dirname(image_path)
    filename = os.path.basename(image_path)

    image = Image.open(image_path).convert("RGB")
    for augmentation_name, augmentation_effect in augmentations.items():
        imaged_augmented = augmentation_effect(image)

        new_image_name = f"{filename}_{augmentation_name}.JPG"
        output_path = os.path.join(directory_name, new_image_name)
        imaged_augmented.save(output_path)
        plt.imshow(imaged_augmented)
        plt.title(augmentation_name)
        # plt.show()


# ============================= Folder Augmentation ===============================

def folder_stats(folder_path: str) -> Dict:
    """
        Returns the number of images to be augmented in every folders based on 
        max value.
        input: path of the main folder
        output: dict containing augmentation plan
    """
    stats = stats_from_dir(folder_path)
    target = max(stats.values())

    augmentation_plan= {}
    for name, current_value in stats.items():
        if current_value < target:
            needed = target - current_value
            images_to_augment = needed // 6
            augmentation_plan[name] = images_to_augment
            # print(f"\n  Augment: {images_to_augment} images ({images_to_augment * 6} new ones) for {name}")
    return augmentation_plan


def folder_augmentation(folder_path: str) -> None:    
    """
        Main function to operate data_augmentation in the required folder.
        input: folder path
        output: None
    """
    augmentation_plan = folder_stats(folder_path)

    for name, value in augmentation_plan.items():
        folder_name = f"{folder_path}/{name}"
        print(folder_name)
        if Path(folder_name).is_dir():
            all_images = [f for f in os.listdir(folder_name) if f.endswith(('.jpg', '.JPG', '.jpeg'))]
            selected_images = random.sample(all_images, min(value, len(all_images)))
            for image in selected_images:
                image_path = os.path.join(folder_name, image)
                print(image_path)
                file_augmentation(image_path)
        print(f"--> Folder {folder_name} has been augmented of {value} images !")

# ===================================== MAIN ======================================

def main(parsed_args):
    
    try:
        if parsed_args.image_path:
            if not parsed_args.image_path.lower().endswith((".jpg", ".jpeg")):
                print("Not a valid image format!")
                return
            file_augmentation(parsed_args.image_path)
            
        elif parsed_args.folder_path:
            if not Path(parsed_args.folder_path).is_dir():
                print("Not a directory!")
                return
            folder_augmentation(parsed_args.folder_path)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str,
                        default=None,
                        help="Selected image for a 6 ways data-augmentation.")
    parser.add_argument('--folder_path',
                        type=str,
                        default=None,
                        help="Option to be used in the training part.")
    parsed_args = parser.parse_args()
    main(parsed_args)

# python Augmentation.py --folder_path ../images_test/Apple
# python Augmentation.py --image_path ../images_test/Grape/Grape_healthy/image\ \(1\).JPG
import argparse, os, random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from plantcv import plantcv as pcv
import numpy as np

def convert_to_grayscale(img: np.array) -> np.array:
    """
        Converts to gray.
    """
    gray_img = pcv.rgb2gray_lab(img, 'a')
    return gray_img


def gaussian_blur(img: np.array) -> np.array:
    """
        Apply some noise on the image (5, 5) is a medium option.
    """
    blur_img = pcv.filters.gaussian_blur(img, ksize=(5, 5), sigma_x=0, sigma_y=None)
    plt.imshow(blur_img)
    plt.title("Gaussian Blur")
    plt.axis('off')
    plt.show()
    return blur_img


def create_mask(gray: np.array) -> np.array:
    """
        Create a mask, a mask is a binary file to be applied later in more
        complex transformations. It allows to isolate the subject from the backplan.
    """
    mask = pcv.threshold.binary(gray_img=gray, threshold=120, object_type='dark') # on met un seuil
    mask = pcv.fill(mask, size=200) # on fill les trous ici 
    mask_clean = pcv.fill_holes(mask)
    plt.imshow(mask_clean, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    plt.show()
    return mask_clean


def apply_mask(img: np.array, mask: np.array) -> np.array:
    """
        Apply the mask.
    """
    masked = pcv.apply_mask(img, mask, mask_color="white")
    plt.imshow(masked)
    plt.title("Masked Image")
    plt.axis('off')
    plt.show()
    return masked


def ROI_objects(img: np.array, mask_img: np.array) -> np.array:
    """
        Find and draw regions of interest on the image.
    """
    objects, hierarchy = pcv.find_objects(img, mask_img)
    if len(objects):
        roi_img = pcv.draw_contours(img, objects, -1, (0, 255, 0), 2)
    else:
        roi_img = img.copy()
    plt.imshow(roi_img)
    plt.title("ROI Objects")
    plt.axis('off')
    plt.show()
    return roi_img


def analyze_object_shape(img: np.array, mask_img: np.array) -> dict:
    """
        Analyze and visualize object shape characteristics.
    """
    analysis_data = pcv.analyze_object(img, mask_img)
    print("Shape Analysis:", analysis_data)
    return analysis_data


def color_histogram(img: np.array, mask_img: np.array) -> dict:
    """
        Generate color histogram analysis.
    """
    hist_data = pcv.analyze_color(img, mask_img, colorspace='hsv')
    print("Color Analysis:", hist_data)
    return hist_data


# ============================== File Transformation ================================


def file_transformation(img_path: str) -> None:
    """
        Generate 6 image transformations based on an original one.
    """
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    
    gray = convert_to_grayscale(img)
    mask = create_mask(gray)
    blur = gaussian_blur(img)
    masked = apply_mask(img, mask)
    roi = ROI_objects(img, mask)
    shape_data = analyze_object_shape(img, mask)
    color_data = color_histogram(img, mask)
    # landmarks = pseudolandmarks(img, mask)
    # plot_all_results([img, blur, mask, roi, shape_data, color_histo, landmarks])

# ============================= Folder Augmentation ===============================

def folder_transformation() -> None:
    pass


# ===================================== MAIN ======================================

def main(parsed_args):
    
    try:
        if parsed_args.image_path:
            if not parsed_args.image_path.lower().endswith((".jpg", ".jpeg")):
                print("Not a valid image format!")
                return
            file_transformation(parsed_args.image_path)
            
        elif parsed_args.folder_path:
            if not Path(parsed_args.folder_path).is_dir():
                print("Not a directory!")
                return
            folder_transformation(parsed_args.folder_path)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str,
                        default=None,
                        help="Selected image for a 6 ways data-transformation.")
    parser.add_argument('--folder_path',
                        type=str,
                        default=None,
                        help="Option to be used in the training part.")
    parsed_args = parser.parse_args()
    main(parsed_args)

# python Transformation.py --image_path ../images_test/Grape/Grape_healthy/image\ \(1\).JPG
# python Transformation.py --folder_path ../images_test/Apple
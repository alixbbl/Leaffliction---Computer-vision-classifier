import argparse, os, random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from plantcv import plantcv as pcv
import numpy as np, cv2, json
from config import OUTPUT_DIR

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


def analyze_object_shape(img: np.array, mask_img: np.array) -> Dict:
    """
        Analyze and visualize object shape characteristics.
    """
    analysis_data = pcv.analyze_object(img, mask_img)
    print("Shape Analysis:", analysis_data)
    return analysis_data


def draw_shape(img: np.array, contours: List[np.array]) -> np.array:
    """
        Draw shape on the original image.
        input : landmarks data, img
        output: numpy array of the new imge
    """
    img_copy = img.copy()
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), thickness=2)
    img_annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_annotated


def extract_landmarks(img: np.array, mask: np.array) -> dict:
    """
        Generate pseudolandmarks on the image using the mask.
        input: original img and mask
        output: landmarks data (dict)
    """
    pseudo_obj, results = pcv.pseudolandmarks(mask, masked_image=img)

    landmark_vis = pseudo_obj['img']
    plt.imshow(landmark_vis)
    plt.title("Pseudolandmarks")
    plt.axis('off')
    plt.show()
    return results  # contient les coordonnées des landmarks


def draw_landmarks(img: np.array, landmarks: List[Dict[str, int]]) -> np.array:
    """
        Draw landmarks on the original image.
        input : landmarks data, img
        output: numpy array of the new imge
    """
    img_copy = img.copy()
    for point in landmarks:
        x, y = point['x'], point['y']
        cv2.circle(img_copy, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    return img_copy

# ============================== File Transformation ================================


def save_images(list_img: List[Any], img_path: Path) -> None :
    """
        Returns in OUTPUT directory the images of the transformation file phase.
        input: Path
        output: None
    """
    basename = img_path.stem
    for idx, img in enumerate(list_img):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = OUTPUT_DIR / f"{basename}_{idx}.png"
        cv2.imwrite(filename, img_bgr)
        print(f"Saved {filename}")

def save_data(shape_data: Dict[Any]) -> None:
    """
        Save a JSON dumps based on the shape_data dict.
        input: dict
        output: None 
    """
    filename = OUTPUT_DIR / "shape_data.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(shape_data, indent=4, ensure_ascii=False)
    print(f"Printed {filename} is ready in {OUTPUT_DIR} !")


def file_transformation(img_path: Path) -> None:
    """
        Data transformation worflow.
        Must start with grayscale, then Mask. Mask is used in the following steps.
        input: path of the original image.
    """
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    
    gray = convert_to_grayscale(img)
    mask = create_mask(gray)
    masked = apply_mask(img, mask)

    blurred = gaussian_blur(masked)
    roi = ROI_objects(masked, mask)
    
    shape_data = analyze_object_shape(masked, mask)
    objects, _ = pcv.find_objects(masked, mask)  # récupérer les contours
    
    shape_img = draw_shape(img, objects)
    landmarks_data = extract_landmarks(masked, mask)
    landmarks_img = draw_landmarks(img, landmarks_data)

    save_images([blurred, masked, roi, shape_img, landmarks_img], img_path)
    save_data(shape_data)


# ============================= Folder Augmentation ===============================


def color_histogram(img: np.array, mask_img: np.array) -> dict:
    """
        Generate color histogram analysis.
    """
    hist_data = pcv.analyze_color(img, mask_img, colorspace='hsv')
    print("Color Analysis:", hist_data)
    return hist_data


def folder_transformation() -> None:
    pass


# ===================================== MAIN ======================================


def main(parsed_args):
    
    try:
        if parsed_args.image_path:
            img_path = Path(parsed_args.image_path)
            if not img_path.suffix.lower() in [".jpg", ".jpeg"]:
                print("Not a valid format !")
                return
            file_transformation(img_path)
            
        elif parsed_args.folder_path:
            folder_path = Path(parsed_args.folder_path)
            if not folder_path.is_dir():
                print("Not a directory !")
                return
            folder_transformation(folder_path)
            
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
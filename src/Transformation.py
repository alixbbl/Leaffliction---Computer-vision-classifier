import argparse, os, random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, List, Dict
from plantcv import plantcv as pcv
import numpy as np, cv2, json
from config import OUTPUT_DIR

def show_image(img: np.array, title: str) -> None:
    """ 
        Displays the image of the transformation.
        input: original image numpy array and title.
        output: None
    """
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def convert_to_grayscale(img: np.array) -> np.array:
    """
        Converts to gray.
        input: original image numpy array.
        output: grayscale numpy array.
    """
    gray_img = pcv.rgb2gray_lab(img, 'a') # Faire un test avec l pendant la correction
    return gray_img


def gaussian_blur(gray: np.array, ksize=5) -> np.array:
    """
        Apply some noise on the image (5, 5) is a medium option.
        input: original image numpy array and ksize, the level of blur applied.
        output: gaussian blurred numpy array.
    """
    blurred = pcv.gaussian_blur(gray, (ksize, ksize), 0)
    return blurred


def create_mask(gray: np.array, threshold=100) -> np.array:
    """
        Create a mask, a mask is a binary file to be applied later in more
        complex transformations. It allows to isolate the subject from the backplan.
    """
    binary = pcv.threshold.gaussian(
        gray_img=gray, ksize=2500, offset=5, object_type="dark"
    )
    binary = pcv.fill(binary, size=50)
    binary_clean = pcv.fill_holes(binary)
    return binary_clean


def apply_mask(img: np.array, mask: np.array) -> np.array:
    """
        Apply the mask on the original image.
        input: original image, mask.
        output: masked image (np array).
    """
    masked = pcv.apply_mask(img, mask, mask_color="white")
    if len(masked.shape) == 2:
        masked = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX)
    return masked


def analyze_object_shape(img: np.array, mask_img: np.array) -> Dict:
    """
        Analyze and visualize object shape characteristics.
        output: shape of the objects contained in the original image.
    """
    shape = pcv.analyze.size(img=img, labeled_mask=mask_img)
    return shape


def ROI_objects(img: np.array, mask_img: np.array) -> np.array:
    """
        Find and draw regions of interest on the image using OpenCV.
        input: original image, mask.
        output: ROI image (np array).
    """
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        roi_img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
    else:
        roi_img = img.copy()
    return roi_img


def draw_shape(img: np.array, contours: List[np.array]) -> np.array:
    """
        Draw shape on the original image.
        input : landmarks data, img
        output: numpy array of the new imge.
    """
    img_copy = img.copy()
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    cv2.drawContours(img_bgr, contours, -1, (0, 255, 0), thickness=2)
    img_annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_annotated


def draw_landmarks(img: np.array, landmarks, color) -> None:
    """
        Draw landmarks on the submitted image.
        input: image and landmarks.
        output : None
    """
    for x, y in landmarks:
        cv2.circle(
            img,
            (int(x), int(y)),
            radius=5,
            color=color,
            thickness=-1,
        )

def apply_landmarks(img:np.array, mask: np.array) -> np.array:
    """
        Extract all landmarks from the image and set the colors for drawing.
        input:
        output: 
    """
    pcv.params.sample_label = "plant"
    pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)

    bottom_landmarks = pcv.outputs.observations["plant"]["bottom_lmk"]["value"]
    top_landmarks = pcv.outputs.observations["plant"]["top_lmk"]["value"]
    center_landmarks = pcv.outputs.observations["plant"]["center_v_lmk"][
        "value"
    ]
    img_with_landmarks = img.copy()
    colors = {
        "bottom": (0, 0, 255),  # Red
        "top": (255, 0, 0),  # Blue
        "center": (0, 255, 0),  # Green
    }
    draw_landmarks(img_with_landmarks, bottom_landmarks, colors["bottom"])
    draw_landmarks(img_with_landmarks, top_landmarks, colors["top"])
    draw_landmarks(img_with_landmarks, center_landmarks, colors["center"])
    return img_with_landmarks

# ============================== File Transformation ================================


def save_transformations(transformations_dict: Dict, directory, img_path: Path) -> None:
    """
    """
    basename = img_path.stem
    print(basename)
    for name, img in transformations_dict.items():
        filename = f"{basename}_{name}.JPG"
        new_image_path = os.path.join(OUTPUT_DIR, filename)
        img = Image.fromarray(img)
        img.save(new_image_path)
        print(f"SAVED: {new_image_path}")


def file_transformation(img_path: Path) -> Dict:
    """
        Data transformation worflow.
        Must start with grayscale, then Mask. Mask is used in the following steps.
        input: path of the original image.
    """
    transformations = {}
    img, imgpath, imgname = pcv.readimage(img_path)

    transformations['original'] = img
    transformations['grayscale'] = convert_to_grayscale(img)
    transformations['gaussian_grayscale'] = gaussian_blur(transformations['grayscale'])
    transformations['mask'] = create_mask(transformations['gaussian_grayscale'])
    transformations['masked'] = apply_mask(transformations['gaussian_grayscale'], transformations['mask'])
    transformations["analyze"] = analyze_object_shape(img, transformations["mask"])
    transformations["roi"] = ROI_objects(img, transformations["mask"])
    transformations["landmarks"] = apply_landmarks(img, transformations["mask"])

    # for name, transformation in transformations.items():
    #     show_image(transformation, title=name)
    
    return transformations


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
            transform_dict = file_transformation(img_path)
            save_transformations(transform_dict, directory=OUTPUT_DIR, img_path=img_path)
            
        elif parsed_args.folder_src and parsed_args.folder_dst:
            folder_src = Path(parsed_args.folder_src)
            folder_dst = Path(parsed_args.folder_dst)
            if not folder_src.is_dir() or not folder_dst.is_dir():
                print("Not a directory !")
                return
            folder_transformation(folder_src)
            # save_transformations_in_dest(folder_dst)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str,
                        default=None,
                        help="Selected image for a 6 ways data-transformation.")
    parser.add_argument('--folder_src',
                        type=str,
                        default=None,
                        help="Source folder path -- Option to be used for the training part.")
    parser.add_argument('--folder_dst',
                        type=str,
                        default=OUTPUT_DIR,
                        help="Destination folder path -- Option to be used for the training part.")
    parsed_args = parser.parse_args()
    main(parsed_args)

# python Transformation.py --image_path ../images_test/Grape/Grape_healthy/image_tst.JPG
# python Transformation.py --folder_src ../images_test/Apple --folder_dst 
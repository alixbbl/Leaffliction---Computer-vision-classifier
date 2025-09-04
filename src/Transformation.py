import argparse
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List
from plantcv import plantcv as pcv
import numpy as np
import cv2
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
    gray_img = pcv.rgb2gray_lab(img, 'a')
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
        complex transformations. It allows to isolate the subject from the
        backplan.
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
        Detects the contours of the main objects.
        input: original image, mask.
        output: ROI image (np array).
    """
    contours, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
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


def apply_landmarks(img: np.array, mask: np.array) -> np.array:
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


def extended_color_histogram(img: np.array, mask: np.array, show: bool):
    """
        Produce an extended color histogram.
        input: original image and the mask
        output: none, displays a complete pixels histogram
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    colors_rgb = ['blue', 'green', 'red']
    for i, color in enumerate(colors_rgb):
        hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, label=color.capitalize())
    plt.title('RGB Channels')
    plt.legend()

    plt.subplot(2, 2, 2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colors_hsv = ['purple', 'cyan', 'orange']  # H, S, V
    labels_hsv = ['Hue', 'Saturation', 'Value']
    for i, (color, label) in enumerate(zip(colors_hsv, labels_hsv)):
        hist = cv2.calcHist([hsv], [i], mask, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, label=label)
    plt.title('HSV Channels')
    plt.legend()

    plt.subplot(2, 2, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    colors_lab = ['gray', 'magenta', 'yellow']  # L, A, B
    labels_lab = ['Lightness', 'Green-Magenta', 'Blue-Yellow']
    for i, (color, label) in enumerate(zip(colors_lab, labels_lab)):
        hist = cv2.calcHist([lab], [i], mask, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7, label=label)
    plt.title('LAB Channels')
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()


# =========================== File Transformation ============================


def file_transformation(img_path: Path, show) -> Dict:
    """
        Data transformation worflow.
        Must start with grayscale, then mask.
        Mask is used in the following steps.
        input: path of the original image.
    """
    transformations = {}
    img, imgpath, imgname = pcv.readimage(img_path)

    transformations['original'] = img
    transformations['grayscale'] = convert_to_grayscale(img)
    transformations['gaussian_grayscale'] = gaussian_blur(
        transformations['grayscale']
    )
    transformations['mask'] = create_mask(
        transformations['gaussian_grayscale']
    )
    transformations['masked'] = apply_mask(
        transformations['gaussian_grayscale'],
        transformations['mask']
    )
    transformations["analyze"] = analyze_object_shape(
        img, transformations["mask"]
    )
    transformations["roi"] = ROI_objects(img, transformations["mask"])
    transformations["landmarks"] = apply_landmarks(
        img, transformations["mask"]
    )
    transformations["color_hist"] = extended_color_histogram(
        img, transformations["mask"], show
    )

    for name, transformation in transformations.items():
        if name != "color_hist" and show:
            show_image(transformation, title=name)

    return transformations


# =========================== Folder Augmentation ============================


def save_transformations(transformations_dict: Dict,
                         directory: Path, img_path: Path) -> None:
    """
        Save all transformations applied on the images in a specified
        directory.
        input:
        output: None
    """
    basename = img_path.stem

    for name, img in transformations_dict.items():
        if isinstance(img, np.ndarray):
            filename = f"{basename}_{name}.JPG"
            new_image_path = os.path.join(directory, filename)
            img = Image.fromarray(img)
            img.save(new_image_path)
            print(f"SAVED: {new_image_path}")


def folder_transformation(folder_src: Path, folder_dst: Path) -> None:
    """
        Apply all the previous transformations on all the images of the
        specified directory.
        input: src and dest paths
        output: None
    """
    authorized_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                             '.png', '.PNG']
    for file_path in folder_src.rglob('*'):
        if file_path.is_file() and file_path.suffix in authorized_extensions:
            print(f"Transformation launched on {file_path.name}")
            transformations = file_transformation(file_path, show=False)
            save_transformations(transformations,
                                 directory=folder_dst, img_path=file_path)


# =================================== MAIN ===================================


def main(parsed_args):
    path = parsed_args.path

    if not Path(path).exists():
        print("Path does not exist!")
        return

    if Path(path).is_file():
        if not path.lower().endswith((".jpg", ".jpeg")):
            print("Not a valid image format!")
            return
        file_transformation(Path(path), show=True)
        print(f"Transformed single image: {path}")

    elif Path(path).is_dir():
        folder_dst = Path(OUTPUT_DIR)
        folder_transformation(Path(path), folder_dst)
        print(f"Transformed folder: {path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=str,
                        help="Path to image file or folder for "
                             "transformation.")
    parsed_args = parser.parse_args()
    main(parsed_args)

# python Transformation.py --image_path
# ../images_test/Grape/Grape_healthy/image_test.JPG
# python Transformation.py --folder_src ../images_test/Apple
# (pour la valeur par defaut)

import argparse, os, random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, List, Dict






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
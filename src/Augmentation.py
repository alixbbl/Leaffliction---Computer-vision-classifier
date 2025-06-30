import argparse
import torchvision.transforms as transforms


def main(parsed_args):
    
    if not parsed_args.image_path.lower().endswith((".jpg", ".jpeg")):
        print("Not an image !")
        return 
    try:

    
    except Exception as e:
        print(f"Error occured during processing directory : {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        type=str,
                        default=None,
                        help="Selected image for a 6 ways data-augmentation.")
    parsed_args = parser.parse_args()
    main(parsed_args)
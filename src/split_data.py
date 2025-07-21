import torch # Framework ML
import torch.nn as nn # Couches du réseau
import torchvision # Outils vision
from torch.utils.data import DataLoader  # Chargement données
from sklearn.model_selection import train_test_split  # Split données
import argparse


def main(parsed_args):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_src',
                        type=str,
                        help="Source folder path.")
    parsed_args = parser.parse_args()
    main(parsed_args)
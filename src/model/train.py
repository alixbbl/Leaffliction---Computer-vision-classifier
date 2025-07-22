import torch # Framework ML
import torch.nn as nn # Couches du réseau
import torchvision # Outils vision
from torch.utils.data import DataLoader  # Chargement données
from sklearn.model_selection import train_test_split  # Split données
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def data_loader(train_folder_path: str, batch_size: int =32) -> DataLoader:
    """
        Creates a dataloader to convert raw images into tensors.
        input: path of the images folder and size of a computed images batch.
        output:
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=train_folder_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader



# ===================================== MAIN ======================================

def main(parsed_args):

    # lancer en premier la distribution
    # faire le test train split => Faire un fichier split independant pour scinder le dataset en 3 parties train/val/test

    # Relancer la distribution 
    # Faire la data augmentation SUR uniquement le dossier Apple_train ou Grape_train
    # 
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_src',
                        type=str,
                        help="Source folder path.")
    parsed_args = parser.parse_args()
    main(parsed_args)
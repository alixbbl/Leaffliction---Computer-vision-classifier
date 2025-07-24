import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from CNN import CNN


def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on a test dataset and display performance metrics.

    This function:
    1. Runs the model on all test images
    2. Collects predictions and true labels
    3. Generates a classification report (precision, recall, F1-score)
    4. Shows a confusion matrix visualization

    Args:
        model: The trained CNN model
        test_loader: DataLoader containing test images
        class_names: List of class names (e.g., ['Apple_Black_rot', ...])
    """
    # Set model to evaluation mode (disables dropout, batch norm updates)
    model.eval()

    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient computation for efficiency (we're not training)
    with torch.no_grad():
        # Process each batch of images
        for images, labels in test_loader:
            # Forward pass: get model predictions
            outputs = model(images)

            # Get the predicted class (highest probability)
            # dim=1 means we take argmax across classes dimension
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and labels (convert to CPU numpy arrays)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert back to tensors for metrics calculation
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    # Generate classification report
    # (shows precision, recall, F1 for each class)
    report = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    print(report)

    # Create confusion matrix (shows prediction errors between classes)
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix as a heatmap
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
    plt.show()


def data_loader(
    folder_path: str, batch_size: int = 32, shuffle=True
) -> DataLoader:
    """Create a dataloader for image data.

    Creates a dataloader to load and convert raw images into tensors.
    All tensors are normalized in the process.

    Args:
        folder_path: Path of the images folder
        batch_size: Size of a computed images batch
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader: Configured dataloader for the dataset
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    return loader


def load_model(weights_path):
    """Load a trained model from saved weights.

    Args:
        weights_path: Path to the .pth file containing model weights

    Returns:
        Model loaded with trained weights, ready for prediction
    """
    # Create a new CNN instance
    model = CNN()

    # Load the saved weights into the model
    model.load_state_dict(torch.load(weights_path))

    # Set to evaluation mode (important for dropout and batch norm)
    model.eval()

    return model


def predict(model, image_path, class_names):
    """Predict the class of a single image.

    Args:
        model: Trained CNN model
        image_path: Path to the image file
        class_names: List of class names for interpretation

    Returns:
        tuple: (original_image, prediction_string)
    """
    # Same transforms as training (must be consistent!)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load and transform the image
    image = Image.open(image_path)

    # Transform and add batch dimension
    # unsqueeze(0) adds dimension: [3, 64, 64] -> [1, 3, 64, 64]
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        # Get model output (logits)
        output = model(image_tensor)

        # Get the predicted class index
        # torch.max returns (values, indices), we want indices
        _, predicted = torch.max(output, 1)

    # Convert index to class name
    prediction = class_names[predicted.item()]

    return image, prediction


def plot_image(image, prediction):
    """Display an image with its prediction as the title.

    Args:
        image: PIL Image object
        prediction: String with the predicted class name
    """
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")  # Hide axes for cleaner display
    plt.show()


def parse_args():
    """Parse command line arguments.

    Expected usage:
    python predict.py <image_or_directory> <weights.pth> --type apple

    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Leaf Disease Prediction")

    # Positional arguments (required)
    parser.add_argument(
        "target", type=str, help="Path to the image file or directory"
    )
    parser.add_argument(
        "weights_path", type=str, help="Path to the model weights file"
    )

    # Named arguments
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["apple", "grape"],
        help="Type of leaf",
    )

    return parser.parse_args()


# Main execution block (only runs if script is executed directly)
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    target = args.target
    weights_path = args.weights_path

    # Validate inputs
    if not os.path.exists(target):
        print(f"Invalid target: {target}")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"Weights file does not exist: {weights_path}")
        sys.exit(1)

    # Define class names for each leaf type
    apple_class_names = [
        "Apple_Black_rot",
        "Apple_healthy",
        "Apple_rust",
        "Apple_scab",
    ]

    grape_class_names = [  # Fixed typo: was "graple"
        "Grape_black_rot",
        "Grape_Esca",
        "Grape_healthy",
        "Grape_spot",
    ]

    # Dictionary to select appropriate class names
    class_names = {
        "apple": apple_class_names,
        "grape": grape_class_names,
    }

    # Load the trained model
    model = load_model(weights_path)

    # Check if target is directory or single file
    if os.path.isdir(target):
        # Directory mode: evaluate on entire test set
        test_loader = data_loader(target, shuffle=False)
        # Evaluate model performance
        evaluate_model(model, test_loader, class_names[args.type])
    else:
        # Single file mode: predict one image
        image, prediction = predict(model, target, class_names[args.type])
        plot_image(image, prediction)

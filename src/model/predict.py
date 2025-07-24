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
        tuple: (original_image, transformed_tensor, prediction_string)
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

    return image, image_tensor, prediction


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


def extract_features(model, image_tensor):
    """Extract feature maps from convolutional layers.

    Args:
        model: Trained CNN model
        image_tensor: Input tensor [1, 3, 64, 64]

    Returns:
        dict: Feature maps from conv1 and conv2 layers
    """
    features = {}

    # Hook to capture activations
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register hooks on convolutional layers
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(image_tensor)

    return features


def plot_with_features(original_image, image_tensor, prediction, model):
    """Display original image and CNN feature maps.

    Args:
        original_image: PIL Image object
        image_tensor: Transformed tensor [1, 3, 64, 64]
        prediction: Predicted class name
        model: Trained CNN model
    """
    # Extract feature maps
    features = extract_features(model, image_tensor)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # Original image
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Transformed input (what CNN receives)
    ax2 = plt.subplot(3, 4, 2)
    transformed_img = image_tensor.squeeze(0).cpu()
    denormalized = transformed_img * 0.5 + 0.5
    denormalized = torch.clamp(denormalized, 0, 1)
    img_display = denormalized.permute(1, 2, 0).numpy()
    ax2.imshow(img_display)
    ax2.set_title("Input to CNN (64x64)")
    ax2.axis("off")

    # Leave positions 3-4 empty for better layout
    for i in range(3, 5):
        ax = plt.subplot(3, 4, i)
        ax.axis("off")

    # Conv1 feature maps (all 6)
    conv1_features = features['conv1'].squeeze(0).cpu()
    for i in range(6):
        ax = plt.subplot(3, 4, 5 + i)
        feature_map = conv1_features[i].numpy()
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Conv1 Filter {i+1}")
        ax.axis("off")

    # Leave positions 11-12 empty for alignment
    for i in range(11, 13):
        ax = plt.subplot(3, 4, i)
        ax.axis("off")

    # Add main title with prediction
    fig.suptitle(
        f"CNN Feature Visualization - Prediction: {prediction}",
        fontsize=16
    )

    # Create second figure for Conv2 features
    fig2 = plt.figure(figsize=(16, 8))
    fig2.suptitle(
        f"Conv2 Feature Maps (16 filters) - Prediction: {prediction}",
        fontsize=16
    )

    # Conv2 feature maps (all 16)
    conv2_features = features['conv2'].squeeze(0).cpu()
    for i in range(16):
        ax = plt.subplot(2, 8, i + 1)
        feature_map = conv2_features[i].numpy()
        ax.imshow(feature_map, cmap='plasma')
        ax.set_title(f"Filter {i+1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_all_conv_filters(original_image, image_tensor, prediction, model):
    """Display all convolutional filters in a comprehensive view.

    Args:
        original_image: PIL Image object
        image_tensor: Transformed tensor [1, 3, 64, 64]
        prediction: Predicted class name
        model: Trained CNN model
    """
    # Extract feature maps
    features = extract_features(model, image_tensor)

    # Create figure with all conv1 and conv2 filters
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle(
        f"All CNN Filters - Prediction: {prediction}",
        fontsize=16
    )

    # Show original image in first position
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis("off")

    # Show input in second position
    transformed_img = image_tensor.squeeze(0).cpu()
    denormalized = transformed_img * 0.5 + 0.5
    denormalized = torch.clamp(denormalized, 0, 1)
    img_display = denormalized.permute(1, 2, 0).numpy()
    axes[0, 1].imshow(img_display)
    axes[0, 1].set_title("CNN Input", fontsize=10)
    axes[0, 1].axis("off")

    # Hide unused spots in first row
    for i in range(2, 8):
        axes[0, i].axis("off")

    # Conv1 features (6 filters) - row 2
    conv1_features = features['conv1'].squeeze(0).cpu()
    for i in range(6):
        ax = axes[1, i]
        feature_map = conv1_features[i].numpy()
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"Conv1-{i+1}", fontsize=10)
        ax.axis("off")

    # Hide unused conv1 spots
    for i in range(6, 8):
        axes[1, i].axis("off")

    # Conv2 features (16 filters) - rows 3 and 4
    conv2_features = features['conv2'].squeeze(0).cpu()
    for i in range(16):
        row = 2 + i // 8
        col = i % 8
        ax = axes[row, col]
        feature_map = conv2_features[i].numpy()
        ax.imshow(feature_map, cmap='plasma')
        ax.set_title(f"Conv2-{i+1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
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

    grape_class_names = [
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
        image, transformed, prediction = predict(
            model, target, class_names[args.type]
        )
        # Show feature maps from CNN
        plot_with_features(image, transformed, prediction, model)
        # Optionally show all filters
        # plot_all_conv_filters(image, transformed, prediction, model)

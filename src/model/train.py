import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from CNN import CNN


def compute_validation_metrics(model, validation_loader):
    """Compute validation metrics for the model.

    Args:
        model: Neural network model
        validation_loader: DataLoader for validation data

    Returns:
        dict: Dictionary containing loss, f1_score, and accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    validation_metrics = {"loss": 0}
    with torch.no_grad():
        accuracy = MulticlassAccuracy(num_classes=8)
        f1_score = MulticlassF1Score(num_classes=8, average="macro")
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            validation_metrics["loss"] += loss.item()
            f1_score.update(preds, labels)
            accuracy.update(preds, labels)
    validation_metrics["loss"] /= len(validation_loader)
    validation_metrics["f1_score"] = f1_score.compute()
    validation_metrics["accuracy"] = accuracy.compute()
    model.train()
    return validation_metrics


def update_metrics_history(
    model, validation_loader, validation_metrics_history
):
    """Update metrics history with new validation metrics.

    Args:
        model: Neural network model
        validation_loader: DataLoader for validation data
        validation_metrics_history: Dictionary to update with new metrics
    """
    new_validation_metrics = compute_validation_metrics(
        model, validation_loader
    )
    for name, value in new_validation_metrics.items():
        validation_metrics_history[name].append(value)


def early_stopping(
    model_path,
    state_dict,
    validation_accuracy,
    epoch,
    best_accuracy=None,
    best_epoch=None,
    counter=0,
    min_delta=0,
    patience=5
):
    """Implement early stopping logic.

    Args:
        model_path: Path to save the model
        state_dict: Model state dictionary
        validation_accuracy: Current validation accuracy
        epoch: Current epoch number
        best_accuracy: Best accuracy so far
        best_epoch: Epoch with best accuracy
        counter: Counter for patience
        min_delta: Minimum change to qualify as improvement
        patience: Number of epochs to wait before stopping

    Returns:
        tuple: (stop, best_accuracy, best_epoch, counter)
    """
    if best_accuracy is None:
        best_epoch = epoch
        best_accuracy = validation_accuracy
    elif validation_accuracy > best_accuracy - min_delta:
        best_epoch = epoch
        best_accuracy = validation_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            torch.save(state_dict, model_path)
            return True, best_accuracy, best_epoch, counter
    return False, best_accuracy, best_epoch, counter


def log_metrics(validation_metrics_history, train_metrics_history):
    """Log current metrics to console.

    Args:
        validation_metrics_history: Validation metrics dictionary
        train_metrics_history: Training metrics dictionary
    """
    print(f"Training Loss: {train_metrics_history['loss'][-1]}")
    print(f"Validation Loss: {validation_metrics_history['loss'][-1]}")
    print(
        f"Validation F1 Score: "
        f"{validation_metrics_history['f1_score'][-1]}"
    )
    print(
        f"Validation Accuracy: "
        f"{validation_metrics_history['accuracy'][-1]}"
    )


def plot_metrics(validation_history, train_history):
    """Plot training and validation metrics.

    Args:
        validation_history: Dictionary with validation metrics
        train_history: Dictionary with training metrics
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(
        validation_history["loss"], label="Validation Loss", color="blue"
    )
    axs[0].plot(train_history["loss"], label="Training Loss", color="green")
    axs[1].plot(
        validation_history["f1_score"],
        label="Validation F1 Score",
        color="orange",
    )
    axs[1].plot(
        train_history["f1_score"], label="Training F1 Score", color="red"
    )
    axs[2].plot(
        validation_history["accuracy"],
        label="Validation Accuracy",
        color="purple",
    )
    axs[2].plot(
        train_history["accuracy"], label="Training Accuracy", color="brown"
    )
    axs[0].set_title("Loss")
    axs[1].set_title("Validation F1 Score")
    axs[2].set_title("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[2].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("F1 Score")
    axs[2].set_ylabel("Accuracy")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.tight_layout()
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


def train(train_loader, valid_loader, epochs, model_path, patience):
    """Train a CNN model for leaf classification with early stopping.

    Args:
        train_loader: DataLoader containing training data batches
        valid_loader: DataLoader containing validation data batches
        epochs: Maximum number of training epochs
        model_path: Path where the best model weights will be saved
        patience: Patience for early stopping

    Returns:
        tuple: (validation_metrics_history, train_metrics_history)
               Both are dictionaries containing f1_score, loss, and
               accuracy lists
    """
    model = CNN()
    # Define loss function - CrossEntropyLoss combines LogSoftmax and NLLLoss
    # Perfect for multi-class classification (4 leaf categories)
    criterion = nn.CrossEntropyLoss()

    # Variables for tracking best model and early stopping
    best_accuracy = None      # Stores the best validation accuracy achieved
    best_epoch = None         # Epoch number where best accuracy was achieved
    counter = 0               # Counts epochs without improvement

    # Initialize AdamW optimizer (Adam with decoupled weight decay)
    # lr=0.001: learning rate - step size for parameter updates
    # weight_decay=0.001: L2 regularization to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    # Dictionaries to store training history for plotting/analysis
    # Each metric is a list that grows with each epoch
    validation_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}
    train_metrics_history = {"f1_score": [], "loss": [], "accuracy": []}

    # Main training loop - iterate through all epochs
    for epoch in range(1, epochs + 1):

        # Track cumulative loss for this epoch
        running_loss = 0.0

        # Iterate through all batches in training data
        # tqdm creates a progress bar showing batch progress
        for i, data in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"), 0
        ):
            # Unpack batch data
            inputs, labels = data

            # Clear gradients from previous iteration
            # PyTorch accumulates gradients by default, so we need to reset
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(inputs)

            # Calculate loss between predictions and ground truth
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()

            # Update model parameters based on gradients
            optimizer.step()

            # Accumulate loss for monitoring
            running_loss += loss.item()

        # === END OF EPOCH EVALUATION ===

        # Calculate and store metrics on entire training set
        # This evaluates how well the model fits training data
        update_metrics_history(model, train_loader, train_metrics_history)

        # Calculate and store metrics on validation set
        # This evaluates how well the model generalizes to unseen data
        update_metrics_history(
            model, valid_loader, validation_metrics_history
        )

        # Log/display current metrics
        log_metrics(validation_metrics_history, train_metrics_history)

        # Get current model parameters for potential saving
        state_dict = model.state_dict()

        # Check if we should stop training early
        stop, best_accuracy, best_epoch, counter = early_stopping(
            model_path,
            state_dict,
            validation_metrics_history["accuracy"][-1],
            epoch,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            counter=counter,
            patience=patience
        )

        # If early stopping triggered, exit training loop
        if stop:
            print(
                f"Early stopping at epoch {best_epoch} "
                f"with best accuracy {best_accuracy}"
            )
            # Return metrics up to this point
            return validation_metrics_history, train_metrics_history

    # If we completed all epochs without early stopping,
    # save the final model
    torch.save(model.state_dict(), model_path)

    # Return complete training history
    return validation_metrics_history, train_metrics_history


def validate_directories(args):
    """Validate that all required directories exist and are valid.

    Args:
        args: Command line arguments containing directory paths
    """
    # Check train directory
    if not os.path.exists(args.train_folder):
        print(f"Train directory does not exist: {args.train_folder}")
        sys.exit(1)
    if not os.path.isdir(args.train_folder):
        print(f"Train directory is not a valid directory: {args.train_folder}")
        sys.exit(1)
    # Check validation directory
    if not os.path.exists(args.valid_folder):
        print(f"Validation directory does not exist: {args.valid_folder}")
        sys.exit(1)
    if not os.path.isdir(args.valid_folder):
        print(
            f"Validation directory is not a valid directory: "
            f"{args.valid_folder}"
        )
        sys.exit(1)


def main(parsed_args):
    """Main training function."""
    # Validate directories before proceeding
    validate_directories(parsed_args)

    # Load data
    train_loader = data_loader(parsed_args.train_folder)
    valid_loader = data_loader(parsed_args.valid_folder, shuffle=False)

    # Ensure model path file exists
    open(parsed_args.model_path, 'a')

    # Train the model
    validation_history, train_history = train(
        train_loader,
        valid_loader,
        parsed_args.epochs,
        parsed_args.model_path,
        parsed_args.patience
    )

    # Plot training metrics
    plot_metrics(validation_history, train_history)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train a model on image dataset"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('train_folder',
                        type=str,
                        help="train folder path.",
                        default="train_dataset")
    parser.add_argument('valid_folder',
                        type=str,
                        help="valid folder.",
                        default="valid_dataset")
    parser.add_argument('model_path',
                        type=str,
                        help="model path.",
                        default="./")
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs',
                        default=10)
    parser.add_argument('--patience',
                        type=int,
                        help='patience',
                        default=10)
    parsed_args = parser.parse_args()
    main(parsed_args)

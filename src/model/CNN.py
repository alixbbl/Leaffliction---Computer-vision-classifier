import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network for leaf classification.
    Architecture: 2 convolutional layers + 3 fully connected layers
    Designed to classify images into 4 categories.
    """
    
    def __init__(self):
        # Call parent class constructor (nn.Module)
        # Required to properly initialize all PyTorch mechanisms
        super(CNN, self).__init__()
        
        # === CONVOLUTIONAL PART (Feature extraction) ===
        
        # First convolutional layer
        # Input: 3 channels (RGB), Output: 6 feature maps, Kernel: 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        
        # Pooling layer (dimensionality reduction)
        # 2x2 window, divides spatial dimensions by 2
        self.pool = nn.MaxPool2d(2, 2)
        
        # Second convolutional layer
        # Input: 6 channels (conv1 output), Output: 16 feature maps, Kernel: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # === MLP PART (Multi-Layer Perceptron for classification) ===
        
        # First fully connected layer
        # Input: 16*13*13 = 2704 neurons (after flattening)
        # Output: 1024 hidden neurons
        self.fc1 = nn.Linear(16 * 13 * 13, 1024)
        
        # Dropout to prevent overfitting
        # Randomly deactivates 50% of neurons during training
        self.dropout1 = nn.Dropout(0.5)
        
        # Second fully connected layer
        # Input: 1024 neurons, Output: 128 neurons
        self.fc2 = nn.Linear(1024, 128)
        
        # Second dropout (also 50%)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer (final classification)
        # Input: 128 neurons, Output: 4 classes
        self.fc3 = nn.Linear(128, 8)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, H, W]
               where H and W are image height and width (expected 64x64)
        
        Returns:
            Output tensor of shape [batch_size, 4] (logits for 4 classes)
        """
        
        # === FEATURE EXTRACTION ===
        
        # Conv1 + ReLU + Pooling
        # [batch, 3, 64, 64] -> [batch, 6, 60, 60] -> [batch, 6, 30, 30]
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv2 + ReLU + Pooling  
        # [batch, 6, 30, 30] -> [batch, 16, 26, 26] -> [batch, 16, 13, 13]
        x = self.pool(F.relu(self.conv2(x)))
        
        # === TRANSITION TO MLP ===
        
        # Flatten 3D tensor to 1D vector
        # [batch, 16, 13, 13] -> [batch, 2704]
        # The '1' means: flatten from dimension 1 onwards (keep batch dimension)
        x = torch.flatten(x, 1)
        
        # === CLASSIFICATION (MLP) ===
        
        # FC1 + ReLU + Dropout
        # [batch, 2704] -> [batch, 1024]
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Only applied during training
        
        # FC2 + ReLU + Dropout
        # [batch, 1024] -> [batch, 128]
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Only applied during training
        
        # Output layer (no activation here)
        # [batch, 128] -> [batch, 4]
        # Raw logits will be transformed to probabilities by Softmax
        # (typically in CrossEntropyLoss function)
        x = self.fc3(x)
        
        return x

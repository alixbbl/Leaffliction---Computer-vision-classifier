#!/bin/bash

# Script to initialize the project environment
# Automatically adapts requirements.txt for the current OS and installs dependencies

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "📱 Operating system detected: $OS"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Update requirements.txt based on OS
if [ "$OS" = "macOS" ]; then
    echo " Updating requirements.txt for macOS..."
    cp util/requirements_macos.txt requirements.txt
    
elif [ "$OS" = "Linux" ]; then
    echo "Updating requirements.txt for Linux..."
    cp util/requirements_linux.txt requirements.txt
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
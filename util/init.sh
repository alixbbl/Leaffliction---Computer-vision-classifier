#!/bin/bash

# Script to initialize the project environment
# Automatically adapts requirements.txt for the current OS and installs dependencies

# Detect OS
if command -v uname >/dev/null 2>&1; then
    OS_NAME=$(uname -s)
    if [[ "$OS_NAME" == "Darwin" ]]; then
        OS="macOS"
    elif [[ "$OS_NAME" == "Linux" ]]; then
        OS="Linux"
    else
        echo "Unsupported operating system: $OS_NAME"
        exit 1
    fi
elif [[ -n "$OSTYPE" ]]; then
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
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
else
    echo "Error: Virtual environment activation script not found"
    exit 1
fi


# Update requirements.txt based on OS
if [ "$OS" = "macOS" ]; then
    echo "Updating requirements.txt for macOS..."
    cp util/requirements_macos.txt requirements.txt
    
elif [ "$OS" = "Linux" ]; then
    echo "Updating requirements.txt for Linux..."
    cp util/requirements_linux.txt requirements.txt
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
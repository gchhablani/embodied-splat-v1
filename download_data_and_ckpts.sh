#!/bin/bash
# This script installs/updates HF CLI, logs in, and downloads the datasets required for the project.
# It creates a 'data' directory in the current working directory and downloads
# the necessary datasets into it.

# Create the data directory if it doesn't exist
mkdir -p data
cd data

# Install or update HF CLI
echo "Installing/updating Hugging Face CLI..."
pip install --upgrade "huggingface_hub[cli]"

# Ensure Git LFS is installed
echo "Setting up Git LFS..."
git lfs install

# Check if already logged in to Hugging Face
echo "Checking Hugging Face login status..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "Already logged in to Hugging Face."
else
    echo "Logging in to Hugging Face..."
    # This will prompt for a token (or use HF_TOKEN env variable if set)
    huggingface-cli login
fi

# Download the dataset using HF CLI
echo "Downloading dataset repository..."
huggingface-cli download gchhablani/embodied-splat --repo-type dataset --local-dir .

# Move ckpts directory outside of data if it exists
if [ -d "ckpts" ]; then
    echo "Moving ckpts directory outside of data..."
    mv ckpts ../ckpts
    echo "Moved ckpts to project root."
fi

cd ..
echo "Done!"

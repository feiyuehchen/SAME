#!/bin/bash
# Script to download TitaNet-small model from NVIDIA NGC

echo "========================================"
echo "Downloading TitaNet-small from NGC"
echo "========================================"

# Check if ngc is installed
if ! command -v ngc &> /dev/null; then
    echo "NGC CLI not found. Installing..."
    pip install ngc-cli
fi

# Create temporary directory
mkdir -p /tmp/titanet_download
cd /tmp/titanet_download

# Download model
echo "Downloading TitaNet-small v1.19.0..."
ngc registry model download-version nvidia/nemo/titanet_small:1.19.0

# Find the .nemo file
NEMO_FILE=$(find . -name "*.nemo" -type f | head -n 1)

if [ -z "$NEMO_FILE" ]; then
    echo "ERROR: .nemo file not found after download"
    exit 1
fi

# Copy to project directory
PROJECT_DIR="/home/feiyueh/hw/titanet_asvspoof2019"
cp "$NEMO_FILE" "$PROJECT_DIR/titanet_small.nemo"

echo "Successfully downloaded TitaNet model to:"
echo "$PROJECT_DIR/titanet_small.nemo"

# Cleanup
cd ~
rm -rf /tmp/titanet_download

echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test the model: python test_forward.py"
echo "2. Start training: python train.py"


#!/bin/bash

# Directory to store the data
DATA_DIR="data/mnist"
mkdir -p "$DATA_DIR"

# MNIST URLs (Amazon S3 Mirror)
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

echo "Downloading MNIST dataset..."

for FILE in "${FILES[@]}"; do
    if [ ! -f "$DATA_DIR/${FILE%.gz}" ]; then
        echo "Downloading $FILE..."
        # Using -L to follow redirects if any
        curl -L -o "$DATA_DIR/$FILE" "$BASE_URL/$FILE"
        
        echo "Extracting $FILE..."
        gunzip -f "$DATA_DIR/$FILE"
    else
        echo "$FILE already exists and is extracted."
    fi
done

echo "MNIST dataset is ready in $DATA_DIR"

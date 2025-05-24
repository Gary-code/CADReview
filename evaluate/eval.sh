#!/bin/bash

SRC_DIRS=(
    "./llava_ov"

)

for SRC_DIR in "${SRC_DIRS[@]}"
do
    echo "Processing $SRC_DIR..."
    python get_ply.py \
        --src "$SRC_DIR"

    python metric.py \
        --src "$SRC_DIR"
    echo "Finished processing $SRC_DIR"
    echo "------------------------"
done
    

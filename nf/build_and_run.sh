#!/bin/bash

# Exit if any command fails
set -e

# Build the project
echo "Building the project..."

LIBRARY_PATH=../bindings/c cargo build

# Check if build succeeded and run the project
if [ $? -eq 0 ]; then
    echo "Build succeeded. Running the project..."
    sudo LD_LIBRARY_PATH=../bindings/c/ ./target/debug/nf $@
else
    echo "Build failed."
fi
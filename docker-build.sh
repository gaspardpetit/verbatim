#!/bin/bash
# Check if the environment variable is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN environment variable is not set."
    exit 1
fi

echo docker build --build-arg="HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN" -t verbatim .
docker build --build-arg="HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN" -t verbatim .


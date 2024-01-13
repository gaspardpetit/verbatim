#!/bin/bash
# Check if the environment variable is set
if [ -z "$TOKEN_HUGGINGFACE" ]; then
    echo "Error: TOKEN_HUGGINGFACE environment variable is not set."
    exit 1
fi

echo docker build --build-arg="TOKEN_HUGGINGFACE=$TOKEN_HUGGINGFACE" -t verbatim .
docker build --build-arg="TOKEN_HUGGINGFACE=$TOKEN_HUGGINGFACE" -t verbatim .


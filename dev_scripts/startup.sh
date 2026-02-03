#!/bin/bash

# Prompt for HF_TOKEN
read -p "Enter your HF_TOKEN: " HF_TOKEN
export HF_TOKEN

# Install uv
pip install uv

# Navigate to lda directory
cd /root/lda

# Sync dependencies
uv sync

# Run the server
uv run python server.py
#!/bin/bash

# Install uv
pip install uv

# Navigate to lda directory
cd /root/lda

# Sync dependencies
uv sync

# Enable HF transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run the server
uv run python server.py
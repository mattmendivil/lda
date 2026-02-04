#!/bin/bash

# Install uv
pip install uv

# Navigate to lda directory
cd /root/lda

# Sync dependencies
uv sync

# Run the server
uv run python server.py
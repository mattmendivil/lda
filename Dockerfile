FROM runpod/pytorch:2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install uv via pip
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --frozen

# Copy application code
COPY . .

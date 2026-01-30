FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install SSH server
RUN apt-get update && apt-get install -y \
    openssh-server \
    && mkdir -p /var/run/sshd \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

WORKDIR /app

# Install uv via pip
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Copy and set up start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose SSH port
EXPOSE 22

# Start SSH and keep container running
CMD ["/start.sh"]

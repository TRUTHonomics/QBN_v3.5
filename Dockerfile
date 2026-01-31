# QBN (QuantBayes Nexus) - GPU Container
# ========================================
# Docker container voor Bayesian Network inference met GPU support
# Gebaseerd op KFL_backend_GPU_v4 structuur

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Metadata
LABEL maintainer="KlineFuturesLab Development Team"
LABEL version="1.0.0"
LABEL description="GPU environment for QuantBayes Nexus Bayesian inference"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    wget \
    git \
    postgresql-client \
    libpq-dev \
    pkg-config \
    vim \
    nano \
    htop \
    tree \
    # Docker CLI voor orchestratie
    docker.io \
    # Additional development tools
    cmake \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# REASON: Split installation to handle large GPU packages with timeouts
# Step 1: Core packages
RUN pip install --default-timeout=300 --retries=5 --no-cache-dir \
    numpy scipy pandas scikit-learn statsmodels networkx pgmpy psycopg2-binary asyncpg python-dotenv pyyaml \
    --break-system-packages

# Step 2: Large GPU packages (torch, cupy) separately with extended timeout
RUN pip install --default-timeout=600 --retries=5 --no-cache-dir \
    torch torchvision torchaudio cupy-cuda12x \
    --break-system-packages

# Step 3: Remaining packages
RUN pip install --default-timeout=300 --retries=5 --no-cache-dir \
    -r requirements.txt \
    --break-system-packages

# Copy application code
COPY . .

# Copy launcher script
COPY start /usr/local/bin/start
RUN chmod +x /usr/local/bin/start

# Create logs directory
RUN mkdir -p /app/_log && chmod 755 /app/_log

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Non-root user for security
RUN useradd -m qbnuser && chown -R qbnuser:qbnuser /app

# REASON: Entrypoint draait als root om database connectie te testen
# De gebruiker kan later naar qbnuser switchen indien gewenst
# USER qbnuser

# Expose port voor FastAPI (toekomstig)
EXPOSE 8081

# Entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - interactive bash
CMD ["/bin/bash"]


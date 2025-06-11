FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

#Update and install basic dependencies including Python
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

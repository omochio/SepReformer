FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

#Update and install basic dependencies including Python
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

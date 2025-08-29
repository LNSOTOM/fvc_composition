# Multi-stage Dockerfile for fvcCOVER - Jetson Deployment
# Stage 1: Base Jetson image with CUDA and JetPack
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 AS jetson-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    libspatialindex-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libblosc-dev \
    liblzma-dev \
    libbz2-dev \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python environment setup
FROM jetson-base AS python-env

# Upgrade pip and install wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install core scientific computing packages optimized for ARM64
RUN pip3 install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.5 \
    seaborn==0.12.2 \
    tqdm==4.66.4 \
    pyyaml==6.0.1

# Install geospatial packages
RUN pip3 install --no-cache-dir \
    rasterio==1.3.9 \
    geopandas==0.14.4 \
    shapely==2.0.4 \
    pyproj==3.6.1 \
    fiona==1.9.6

# Install computer vision packages
RUN pip3 install --no-cache-dir \
    opencv-python==4.8.1.78 \
    pillow==10.4.0 \
    scikit-image==0.21.0 \
    albumentations==1.3.1

# Install PyTorch ecosystem (using pre-built Jetson wheels)
RUN pip3 install --no-cache-dir \
    torchvision==0.15.2 \
    torchaudio==2.0.2

# Install additional ML/DL packages
RUN pip3 install --no-cache-dir \
    torchmetrics==1.2.1 \
    pytorch-lightning==2.1.4

# Stage 3: Specialized geospatial tools
FROM python-env AS geospatial-tools

# Install PDAL for point cloud processing
RUN apt-get update && apt-get install -y \
    pdal \
    libpdal-dev \
    python3-pdal \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for point cloud processing
RUN pip3 install --no-cache-dir \
    open3d==0.17.0 \
    laspy==2.5.4

# Stage 4: Final application stage
FROM geospatial-tools AS fvc-app

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs

# Install project-specific requirements
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Set environment variables for the application
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PROJ_LIB=/usr/share/proj
ENV GDAL_DATA=/usr/share/gdal

# Create a non-root user for security
RUN useradd -m -u 1000 jetson && \
    chown -R jetson:jetson /app
USER jetson

# Expose ports if needed
EXPOSE 8888 6006

# Default command
CMD ["python3", "-c", "print('fvcCOVER container ready for Jetson deployment')"]
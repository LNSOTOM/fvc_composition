# fvcCOVER Jetson Deployment Guide

This guide provides comprehensive instructions for containerizing and deploying the fvcCOVER application on NVIDIA Jetson devices for edge AI inference.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Container Options](#container-options)
5. [Deployment Modes](#deployment-modes)
6. [API Usage](#api-usage)
7. [Model Optimization](#model-optimization)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

## Overview

The fvcCOVER containerization solution provides:

- **Multi-stage Docker builds** optimized for Jetson ARM64 architecture
- **Inference-only containers** with minimal dependencies for edge deployment
- **Development containers** with full capabilities for model training and experimentation
- **RESTful API service** for model inference
- **Model optimization tools** for improved performance on Jetson hardware
- **Automated deployment scripts** for easy setup and management

## Prerequisites

### Hardware Requirements

- NVIDIA Jetson device (Nano, Xavier NX, AGX Xavier, or Orin series)
- Minimum 8GB storage (16GB+ recommended)
- MicroSD card (64GB+ recommended) or NVMe SSD

### Software Requirements

- JetPack 5.0+ (L4T R35.1+)
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Runtime

### Installation

1. **Install JetPack SDK**
   ```bash
   # Follow NVIDIA's official JetPack installation guide
   # https://developer.nvidia.com/embedded/jetpack
   ```

2. **Install Docker**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Install Docker Compose**
   ```bash
   sudo apt-get update
   sudo apt-get install docker-compose-plugin
   ```

4. **Install NVIDIA Container Runtime**
   ```bash
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
   sudo apt-get update
   sudo apt-get install nvidia-container-runtime
   sudo systemctl restart docker
   ```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/LNSOTOM/fvc_composition.git
   cd fvc_composition
   ```

2. **Prepare your data structure**
   ```bash
   mkdir -p data models outputs logs
   # Place your trained models in the models/ directory
   # Place input images in the data/ directory
   ```

3. **Build and run inference service**
   ```bash
   ./docker/deploy.sh build-inference
   ./docker/deploy.sh run-inference
   ```

4. **Test the API**
   ```bash
   curl http://localhost:5000/health
   ```

## Container Options

### Development Container (`Dockerfile`)

Full-featured container for development and training:
- Complete Python environment with all dependencies
- Jupyter Lab for interactive development
- TensorBoard for monitoring
- GDAL and geospatial libraries
- Point cloud processing tools (PDAL, Open3D)

**Build:**
```bash
./docker/deploy.sh build-dev
```

**Run:**
```bash
./docker/deploy.sh run-dev
```

**Access:**
- Jupyter Lab: http://localhost:8888
- TensorBoard: http://localhost:6006

### Inference Container (`Dockerfile.inference`)

Lightweight container optimized for inference:
- Minimal dependencies for fast startup
- Optimized for edge deployment
- RESTful API service
- Model optimization tools

**Build:**
```bash
./docker/deploy.sh build-inference
```

**Run:**
```bash
./docker/deploy.sh run-inference
```

**Access:**
- API: http://localhost:5000

## Deployment Modes

### 1. Interactive Development

For model development and experimentation:

```bash
./docker/deploy.sh run-dev
```

Features:
- Full Python environment
- Jupyter Lab interface
- Volume mounting for persistent data
- GPU acceleration

### 2. REST API Service

For production inference via HTTP API:

```bash
./docker/deploy.sh run-inference
```

Features:
- RESTful endpoints for inference
- Batch processing support
- Health monitoring
- Automatic model loading

### 3. Batch Processing

For processing large datasets:

```bash
./docker/deploy.sh run-batch
```

Features:
- Command-line batch processing
- Large-scale image processing
- Output to mounted volumes

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "device": "cuda:0"
}
```

### Load Model

```bash
curl -X POST http://localhost:5000/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/app/models/best_model.ckpt"}'
```

### Single Image Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/your/image.tif"
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/predict_batch \
  -F "files=@image1.tif" \
  -F "files=@image2.tif" \
  -F "files=@image3.tif"
```

### System Information

```bash
curl http://localhost:5000/system_info
```

## Model Optimization

### Automatic Optimization

The deployment includes automated model optimization for Jetson:

```bash
./docker/deploy.sh optimize --model-path /path/to/model.ckpt
```

Optimizations include:
- **TorchScript tracing** for faster inference
- **FP16 precision** to reduce memory usage
- **Dynamic quantization** for INT8 acceleration
- **CUDA kernel optimization**

### Manual Optimization

For custom optimization workflows:

```python
from docker.jetson_optimizer import JetsonOptimizer

optimizer = JetsonOptimizer()

# Load your model
model = load_your_model()

# Optimize for inference
optimized_model = optimizer.optimize_model_for_inference(model)

# Create TorchScript version
traced_model = optimizer.trace_model(optimized_model, input_shape=(1, 5, 256, 256))

# Save optimized model
torch.jit.save(traced_model, 'optimized_model.pt')
```

## Performance Tuning

### Memory Management

1. **GPU Memory Optimization**
   ```python
   # Enable memory pooling
   torch.cuda.memory._set_allocator_settings("expandable_segments:True")
   
   # Set memory fraction (adjust for your Jetson model)
   torch.cuda.set_per_process_memory_fraction(0.8)
   ```

2. **Swap Configuration**
   ```bash
   # Increase swap for larger models (on Jetson with limited RAM)
   sudo systemctl disable nvzramconfig
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### CPU Optimization

1. **CPU Governor**
   ```bash
   # Set performance mode
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Docker Resource Limits**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 6G
       reservations:
         memory: 4G
   ```

### Monitoring Performance

```bash
# System monitoring
./docker/deploy.sh status

# Model benchmarking
./docker/deploy.sh benchmark

# Container logs
./docker/deploy.sh logs
```

## Environment Variables

Key environment variables for configuration:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all

# Model settings
export MODEL_PATH=/app/models/best_model.ckpt
export BATCH_SIZE=1
export INPUT_SIZE=256

# Performance settings
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or model input size
   # Enable memory optimization
   # Use FP16 precision
   ```

2. **Slow Inference**
   ```bash
   # Check model optimization
   ./docker/deploy.sh benchmark
   
   # Verify TensorRT optimization
   # Monitor GPU utilization
   ```

3. **Container Build Issues**
   ```bash
   # Clear Docker cache
   docker system prune -a
   
   # Check available disk space
   df -h
   
   # Verify base image availability
   docker pull nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
   ```

### Debug Mode

Enable debug logging:

```bash
# Set debug environment
export PYTHONPATH=/app
export CUDA_LAUNCH_BLOCKING=1

# Run with debug output
docker-compose run --rm fvc-inference python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Your debug code here
"
```

### Performance Profiling

```bash
# Profile GPU usage
nvidia-smi dmon

# Profile container resources
docker stats

# Profile Python code
python3 -m cProfile your_script.py
```

## Jetson-Specific Considerations

### Jetson Nano (4GB)
- Use FP16 precision
- Reduce batch size to 1
- Enable swap memory
- Use inference-only container

### Jetson Xavier NX (8GB)
- Standard configuration works well
- Can handle larger batch sizes
- Full development container supported

### Jetson AGX Xavier/Orin (16GB+)
- Full capabilities available
- Can run multiple models
- Suitable for development and production

## Support

For issues specific to this containerization:
1. Check the logs: `./docker/deploy.sh logs`
2. Verify system status: `./docker/deploy.sh status`
3. Run benchmarks: `./docker/deploy.sh benchmark`
4. Open an issue in the repository with system information

For general fvcCOVER questions, refer to the main project documentation and repository.
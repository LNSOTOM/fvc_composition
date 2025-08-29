#!/bin/bash

# FVC Composition - Jetson Deployment Script
# This script helps with building and deploying the containerized application on Jetson devices

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_MODEL_PATH="/app/models/best_model.ckpt"
DEFAULT_DATA_PATH="./data"
DEFAULT_OUTPUT_PATH="./outputs"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "================================================"
    echo "  fvcCOVER - Jetson Deployment Manager"
    echo "================================================"
    echo -e "${NC}"
}

print_usage() {
    echo -e "${YELLOW}Usage: $0 [COMMAND] [OPTIONS]${NC}"
    echo ""
    echo "Commands:"
    echo "  build-dev      Build development container"
    echo "  build-inference Build inference-only container"
    echo "  run-dev        Run development environment"
    echo "  run-inference  Run inference service"
    echo "  run-batch      Run batch inference"
    echo "  optimize       Optimize model for Jetson"
    echo "  benchmark      Benchmark model performance"
    echo "  status         Show system status"
    echo "  logs           Show container logs"
    echo "  stop           Stop all containers"
    echo "  clean          Clean up containers and images"
    echo ""
    echo "Options:"
    echo "  --model-path   Path to model file (default: $DEFAULT_MODEL_PATH)"
    echo "  --data-path    Path to data directory (default: $DEFAULT_DATA_PATH)"
    echo "  --output-path  Path to output directory (default: $DEFAULT_OUTPUT_PATH)"
    echo "  --help         Show this help message"
}

check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime not available. GPU acceleration disabled.${NC}"
    else
        echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
    fi
    
    echo -e "${GREEN}✓ Requirements check completed${NC}"
}

build_development() {
    echo -e "${BLUE}Building development container...${NC}"
    docker-compose build fvc-development
    echo -e "${GREEN}✓ Development container built successfully${NC}"
}

build_inference() {
    echo -e "${BLUE}Building inference container...${NC}"
    docker-compose build fvc-inference
    echo -e "${GREEN}✓ Inference container built successfully${NC}"
}

run_development() {
    echo -e "${BLUE}Starting development environment...${NC}"
    docker-compose up -d fvc-development
    echo -e "${GREEN}✓ Development environment started${NC}"
    echo -e "${YELLOW}Access Jupyter Lab at: http://localhost:8888${NC}"
    echo -e "${YELLOW}Access TensorBoard at: http://localhost:6006${NC}"
}

run_inference() {
    echo -e "${BLUE}Starting inference service...${NC}"
    docker-compose up -d fvc-inference
    echo -e "${GREEN}✓ Inference service started${NC}"
    echo -e "${YELLOW}API available at: http://localhost:5000${NC}"
    echo ""
    echo "Test the API with:"
    echo "curl http://localhost:5000/health"
}

run_batch_inference() {
    echo -e "${BLUE}Running batch inference...${NC}"
    docker-compose run --rm fvc-batch-inference
    echo -e "${GREEN}✓ Batch inference completed${NC}"
}

optimize_model() {
    local model_path=${1:-$DEFAULT_MODEL_PATH}
    echo -e "${BLUE}Optimizing model for Jetson...${NC}"
    
    docker-compose run --rm fvc-inference python3 docker/jetson_optimizer.py \
        --model-path "$model_path" \
        --output-path "/app/models/optimized_model.pt"
    
    echo -e "${GREEN}✓ Model optimization completed${NC}"
}

benchmark_model() {
    echo -e "${BLUE}Benchmarking model performance...${NC}"
    
    docker-compose run --rm fvc-inference python3 -c "
from docker.jetson_optimizer import JetsonOptimizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
optimizer = JetsonOptimizer()

# Show system stats
print('System Stats:')
stats = optimizer.get_system_stats()
for key, value in stats.items():
    print(f'  {key}: {value}')

# Load and benchmark model if available
try:
    from phase_3_models.unet_site_models.model.unet_module import UNetModule
    model = UNetModule().to(optimizer.device)
    model.eval()
    
    print('\\nBenchmark Results:')
    results = optimizer.benchmark_inference(model, (1, 5, 256, 256))
    for key, value in results.items():
        print(f'  {key}: {value}')
except Exception as e:
    print(f'Benchmark failed: {e}')
"
    
    echo -e "${GREEN}✓ Benchmark completed${NC}"
}

show_status() {
    echo -e "${BLUE}Container Status:${NC}"
    docker-compose ps
    
    echo -e "\n${BLUE}System Resources:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n${BLUE}GPU Status:${NC}"
        nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
}

show_logs() {
    local service=${1:-"fvc-inference"}
    echo -e "${BLUE}Showing logs for $service...${NC}"
    docker-compose logs -f "$service"
}

stop_containers() {
    echo -e "${BLUE}Stopping all containers...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ All containers stopped${NC}"
}

clean_up() {
    echo -e "${BLUE}Cleaning up containers and images...${NC}"
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Main script logic
print_header

# Parse command line arguments
MODEL_PATH="$DEFAULT_MODEL_PATH"
DATA_PATH="$DEFAULT_DATA_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        build-dev)
            COMMAND="build-dev"
            shift
            ;;
        build-inference)
            COMMAND="build-inference"
            shift
            ;;
        run-dev)
            COMMAND="run-dev"
            shift
            ;;
        run-inference)
            COMMAND="run-inference"
            shift
            ;;
        run-batch)
            COMMAND="run-batch"
            shift
            ;;
        optimize)
            COMMAND="optimize"
            shift
            ;;
        benchmark)
            COMMAND="benchmark"
            shift
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        logs)
            COMMAND="logs"
            SERVICE="$2"
            shift 2
            ;;
        stop)
            COMMAND="stop"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Execute command
case ${COMMAND:-""} in
    build-dev)
        check_requirements
        build_development
        ;;
    build-inference)
        check_requirements
        build_inference
        ;;
    run-dev)
        check_requirements
        run_development
        ;;
    run-inference)
        check_requirements
        run_inference
        ;;
    run-batch)
        check_requirements
        run_batch_inference
        ;;
    optimize)
        check_requirements
        optimize_model "$MODEL_PATH"
        ;;
    benchmark)
        check_requirements
        benchmark_model
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$SERVICE"
        ;;
    stop)
        stop_containers
        ;;
    clean)
        clean_up
        ;;
    "")
        echo -e "${RED}No command specified${NC}"
        print_usage
        exit 1
        ;;
esac
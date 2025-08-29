# Example scripts for common Jetson deployment scenarios

# 1. Quick deployment for inference
echo "=== Quick Jetson Deployment ==="
./docker/deploy.sh build-inference
./docker/deploy.sh run-inference

# Test the deployment
sleep 10
curl http://localhost:5000/health

# 2. Development environment setup
echo -e "\n=== Development Environment ==="
./docker/deploy.sh build-dev
./docker/deploy.sh run-dev

echo "Jupyter Lab available at: http://localhost:8888"
echo "TensorBoard available at: http://localhost:6006"

# 3. Model optimization workflow
echo -e "\n=== Model Optimization ==="
./docker/deploy.sh optimize --model-path ./models/your_model.ckpt

# 4. Performance benchmarking
echo -e "\n=== Performance Benchmark ==="
./docker/deploy.sh benchmark

# 5. Batch inference example
echo -e "\n=== Batch Processing ==="
./docker/deploy.sh run-batch

# 6. Monitor deployment
echo -e "\n=== Monitoring ==="
./docker/deploy.sh status

# 7. Clean up
echo -e "\n=== Cleanup ==="
# ./docker/deploy.sh stop
# ./docker/deploy.sh clean
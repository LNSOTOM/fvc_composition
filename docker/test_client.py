#!/usr/bin/env python3
"""
Example client for testing the FVC inference API
Demonstrates how to use the containerized service for various inference tasks
"""

import requests
import json
import os
import time
from pathlib import Path
import argparse

class FVCClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check if the service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def load_model(self, model_path):
        """Load a specific model"""
        try:
            data = {"model_path": model_path}
            response = self.session.post(f"{self.base_url}/load_model", json=data)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Model loading failed: {e}")
            return None
    
    def predict_single(self, image_path):
        """Predict on a single image"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Single prediction failed: {e}")
            return None
    
    def predict_batch(self, image_paths):
        """Predict on multiple images"""
        try:
            files = []
            for path in image_paths:
                files.append(('files', open(path, 'rb')))
            
            response = self.session.post(f"{self.base_url}/predict_batch", files=files)
            
            # Close file handles
            for _, file_handle in files:
                file_handle.close()
            
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return None
    
    def get_system_info(self):
        """Get system information"""
        try:
            response = self.session.get(f"{self.base_url}/system_info")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"System info failed: {e}")
            return None

def test_basic_functionality(client):
    """Test basic API functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Health check
    print("1. Health Check...")
    health = client.health_check()
    if health:
        print(f"   ✓ Service is healthy: {health}")
    else:
        print("   ✗ Service is not healthy")
        return False
    
    # System info
    print("2. System Information...")
    sys_info = client.get_system_info()
    if sys_info:
        print(f"   ✓ System info retrieved")
        for key, value in sys_info.items():
            print(f"     {key}: {value}")
    else:
        print("   ✗ Failed to get system info")
    
    return True

def test_model_loading(client, model_path):
    """Test model loading"""
    print("\n=== Testing Model Loading ===")
    
    if not os.path.exists(model_path):
        print(f"   ⚠ Model file not found: {model_path}")
        print("   Using default model path")
        model_path = "/app/models/best_model.ckpt"
    
    result = client.load_model(model_path)
    if result and result.get('success'):
        print(f"   ✓ Model loaded successfully")
        return True
    else:
        print(f"   ✗ Model loading failed: {result}")
        return False

def test_inference(client, test_images_dir):
    """Test inference functionality"""
    print("\n=== Testing Inference ===")
    
    # Find test images
    test_images = []
    if os.path.exists(test_images_dir):
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            test_images.extend(Path(test_images_dir).glob(ext))
    
    if not test_images:
        print(f"   ⚠ No test images found in {test_images_dir}")
        return False
    
    test_images = [str(p) for p in test_images[:3]]  # Limit to 3 images
    print(f"   Found {len(test_images)} test images")
    
    # Test single prediction
    print("1. Single Image Prediction...")
    result = client.predict_single(test_images[0])
    if result and result.get('success'):
        print(f"   ✓ Single prediction successful")
        print(f"     Shape: {result['result']['shape']}")
    else:
        print(f"   ✗ Single prediction failed: {result}")
        return False
    
    # Test batch prediction (if multiple images)
    if len(test_images) > 1:
        print("2. Batch Prediction...")
        result = client.predict_batch(test_images[:2])
        if result and result.get('success'):
            print(f"   ✓ Batch prediction successful")
            print(f"     Processed {len(result['results'])} images")
        else:
            print(f"   ✗ Batch prediction failed: {result}")
            return False
    
    return True

def benchmark_performance(client, test_image, num_runs=10):
    """Benchmark inference performance"""
    print(f"\n=== Performance Benchmark ({num_runs} runs) ===")
    
    if not os.path.exists(test_image):
        print(f"   ⚠ Test image not found: {test_image}")
        return
    
    times = []
    successful_runs = 0
    
    print(f"   Running {num_runs} inference tests...")
    
    for i in range(num_runs):
        start_time = time.time()
        result = client.predict_single(test_image)
        end_time = time.time()
        
        if result and result.get('success'):
            times.append(end_time - start_time)
            successful_runs += 1
        
        if (i + 1) % 5 == 0:
            print(f"   Progress: {i + 1}/{num_runs}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1.0 / avg_time
        
        print(f"   ✓ Benchmark Results:")
        print(f"     Successful runs: {successful_runs}/{num_runs}")
        print(f"     Average time: {avg_time:.3f}s")
        print(f"     Min time: {min_time:.3f}s")
        print(f"     Max time: {max_time:.3f}s")
        print(f"     Average FPS: {fps:.2f}")
    else:
        print(f"   ✗ No successful runs")

def main():
    parser = argparse.ArgumentParser(description="FVC Inference API Test Client")
    parser.add_argument("--url", default="http://localhost:5000", 
                       help="Base URL of the FVC service")
    parser.add_argument("--model-path", default="/app/models/best_model.ckpt",
                       help="Path to model file")
    parser.add_argument("--test-images", default="./data",
                       help="Directory containing test images")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=10,
                       help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Initialize client
    client = FVCClient(args.url)
    
    print(f"FVC Inference API Test Client")
    print(f"Service URL: {args.url}")
    print(f"Model Path: {args.model_path}")
    print(f"Test Images: {args.test_images}")
    
    # Test basic functionality
    if not test_basic_functionality(client):
        print("\n❌ Basic functionality test failed")
        return 1
    
    # Test model loading
    if not test_model_loading(client, args.model_path):
        print("\n❌ Model loading test failed")
        return 1
    
    # Test inference
    if not test_inference(client, args.test_images):
        print("\n❌ Inference test failed")
        return 1
    
    # Run benchmark if requested
    if args.benchmark:
        test_images = list(Path(args.test_images).glob("*.tif"))
        if test_images:
            benchmark_performance(client, str(test_images[0]), args.benchmark_runs)
        else:
            print("\n⚠ No test images found for benchmark")
    
    print("\n✅ All tests completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
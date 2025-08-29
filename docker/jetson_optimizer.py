#!/usr/bin/env python3
"""
Jetson optimization utilities for FVC models
Includes model optimization, memory management, and performance monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import gc
import psutil
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class JetsonOptimizer:
    """Optimization utilities for Jetson deployment"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize PyTorch model for inference on Jetson"""
        try:
            logger.info("Optimizing model for Jetson inference...")
            
            # Set to evaluation mode
            model.eval()
            
            # Fuse operations for better performance
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(model)
            
            # Convert to half precision if supported
            if self.device.type == 'cuda':
                try:
                    model = model.half()
                    logger.info("Model converted to half precision (FP16)")
                except Exception as e:
                    logger.warning(f"Failed to convert to half precision: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def trace_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Optional[torch.jit.ScriptModule]:
        """Create TorchScript traced model for better performance"""
        try:
            logger.info(f"Tracing model with input shape: {input_shape}")
            
            # Create dummy input
            dummy_input = torch.randn(input_shape).to(self.device)
            if self.device.type == 'cuda':
                try:
                    dummy_input = dummy_input.half()
                except Exception:
                    pass
            
            # Trace the model
            model.eval()
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
            
            # Optimize traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            logger.info("Model tracing completed successfully")
            return traced_model
            
        except Exception as e:
            logger.error(f"Model tracing failed: {e}")
            return None
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization for reduced memory usage"""
        try:
            logger.info("Applying dynamic quantization...")
            
            # Apply dynamic quantization to linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization completed")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        try:
            if torch.cuda.is_available():
                # Clear cache
                torch.cuda.empty_cache()
                
                # Set memory fraction if needed (adjust based on your Jetson model)
                # torch.cuda.set_per_process_memory_fraction(0.8)
                
                # Enable memory pooling
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
                
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.warning(f"Memory optimization warning: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        stats = {}
        
        try:
            # CPU stats
            stats['cpu_percent'] = psutil.cpu_percent(interval=1)
            stats['memory_percent'] = psutil.virtual_memory().percent
            stats['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
            
            # GPU stats (if available)
            if torch.cuda.is_available():
                stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                stats['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
                stats['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            
            # Temperature (Jetson specific)
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    stats['cpu_temperature_c'] = temp
            except Exception:
                pass
                
            try:
                with open('/sys/class/thermal/thermal_zone1/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    stats['gpu_temperature_c'] = temp
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
        
        return stats
    
    def benchmark_inference(self, model: nn.Module, input_shape: Tuple[int, ...], 
                          num_runs: int = 100) -> Dict:
        """Benchmark model inference performance"""
        try:
            logger.info(f"Benchmarking inference with {num_runs} runs...")
            
            model.eval()
            dummy_input = torch.randn(input_shape).to(self.device)
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize()
            
            # Benchmark runs
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(dummy_input)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            results = {
                'mean_inference_time_ms': float(np.mean(times) * 1000),
                'std_inference_time_ms': float(np.std(times) * 1000),
                'min_inference_time_ms': float(np.min(times) * 1000),
                'max_inference_time_ms': float(np.max(times) * 1000),
                'fps': float(1.0 / np.mean(times)),
                'num_runs': num_runs
            }
            
            logger.info(f"Benchmark results: {results['mean_inference_time_ms']:.2f}ms avg, {results['fps']:.2f} FPS")
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}

def optimize_model_for_jetson(model_path: str, output_path: str, 
                             input_shape: Tuple[int, ...] = (1, 5, 256, 256)) -> bool:
    """Main function to optimize a model for Jetson deployment"""
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cuda')
        
        # Import model class (adjust based on your model structure)
        from phase_3_models.unet_site_models.model.unet_module import UNetModule
        
        model = UNetModule()
        model.load_state_dict(checkpoint['state_dict'])
        
        # Initialize optimizer
        optimizer = JetsonOptimizer()
        
        # Optimize model
        model = optimizer.optimize_model_for_inference(model)
        
        # Trace model
        traced_model = optimizer.trace_model(model, input_shape)
        
        if traced_model is not None:
            # Save optimized model
            torch.jit.save(traced_model, output_path)
            logger.info(f"Optimized model saved to {output_path}")
            
            # Benchmark performance
            benchmark_results = optimizer.benchmark_inference(traced_model, input_shape)
            logger.info(f"Benchmark results: {benchmark_results}")
            
            return True
        else:
            logger.error("Model tracing failed")
            return False
            
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Optimize a model for Jetson
    model_path = "/app/models/best_model.ckpt"
    output_path = "/app/models/optimized_model.pt"
    
    if optimize_model_for_jetson(model_path, output_path):
        print("Model optimization completed successfully!")
    else:
        print("Model optimization failed!")
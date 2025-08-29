#!/usr/bin/env python3
"""
Flask-based inference service for FVC mapping on Jetson devices
Provides RESTful API for model inference with optimized memory usage
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from PIL import Image
import rasterio
from io import BytesIO
import gc

# Add the app root to Python path
sys.path.append('/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)

class FVCInferenceService:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
    def load_model(self, model_path):
        """Load the trained FVC model"""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model checkpoint
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Import model class (adjust import based on your model structure)
                from phase_3_models.unet_site_models.model.unet_module import UNetModule
                
                # Initialize model
                self.model = UNetModule()
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self.model_loaded = True
                logger.info("Model loaded successfully")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image_data):
        """Preprocess input image for inference"""
        try:
            # Convert to numpy array if needed
            if isinstance(image_data, Image.Image):
                image_array = np.array(image_data)
            else:
                image_array = image_data
            
            # Normalize to [0, 1] range
            if image_array.max() > 1.0:
                image_array = image_array.astype(np.float32) / 255.0
            
            # Convert to torch tensor
            if len(image_array.shape) == 3:
                # Add batch dimension and rearrange to [B, C, H, W]
                tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            else:
                tensor = torch.from_numpy(image_array).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def postprocess_output(self, model_output):
        """Postprocess model output to get FVC map"""
        try:
            # Apply softmax to get class probabilities
            probabilities = torch.softmax(model_output, dim=1)
            
            # Get class predictions
            predictions = torch.argmax(probabilities, dim=1)
            
            # Convert to numpy for further processing
            pred_np = predictions.cpu().numpy().squeeze()
            prob_np = probabilities.cpu().numpy().squeeze()
            
            return pred_np, prob_np
            
        except Exception as e:
            logger.error(f"Error postprocessing output: {str(e)}")
            return None, None
    
    def predict(self, image_data):
        """Run inference on input image"""
        if not self.model_loaded:
            return None, "Model not loaded"
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image_data)
            if input_tensor is None:
                return None, "Failed to preprocess image"
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Postprocess
            predictions, probabilities = self.postprocess_output(output)
            
            # Clean up GPU memory
            del input_tensor, output
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'shape': predictions.shape
            }, None
            
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

# Initialize inference service
inference_service = FVCInferenceService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_service.model_loaded,
        'cuda_available': torch.cuda.is_available(),
        'device': str(inference_service.device)
    })

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load model endpoint"""
    data = request.get_json()
    model_path = data.get('model_path', '/app/models/best_model.ckpt')
    
    success = inference_service.load_model(model_path)
    
    return jsonify({
        'success': success,
        'message': 'Model loaded successfully' if success else 'Failed to load model'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if not inference_service.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 400
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            
            # Read image
            if file.filename.lower().endswith(('.tif', '.tiff')):
                # Handle GeoTIFF files
                with rasterio.open(BytesIO(file.read())) as dataset:
                    image_data = dataset.read()
                    # Rearrange from [C, H, W] to [H, W, C]
                    image_data = np.transpose(image_data, (1, 2, 0))
            else:
                # Handle regular image files
                image = Image.open(BytesIO(file.read()))
                image_data = np.array(image)
        
        elif 'image_data' in request.json:
            # Handle base64 encoded data or direct array
            image_data = np.array(request.json['image_data'])
        
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Run prediction
        result, error = inference_service.predict(image_data)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint for multiple images"""
    try:
        if not inference_service.model_loaded:
            return jsonify({'error': 'Model not loaded'}), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            # Process each file
            if file.filename.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(BytesIO(file.read())) as dataset:
                    image_data = dataset.read()
                    image_data = np.transpose(image_data, (1, 2, 0))
            else:
                image = Image.open(BytesIO(file.read()))
                image_data = np.array(image)
            
            # Run prediction
            result, error = inference_service.predict(image_data)
            
            results.append({
                'filename': file.filename,
                'result': result if not error else None,
                'error': error
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/system_info', methods=['GET'])
def system_info():
    """Get system information"""
    try:
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': str(inference_service.device),
            'model_loaded': inference_service.model_loaded
        }
        
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
            info['cuda_memory_cached'] = torch.cuda.memory_reserved(0)
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Auto-load model if available
    default_model_path = '/app/models/best_model.ckpt'
    if os.path.exists(default_model_path):
        inference_service.load_model(default_model_path)
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
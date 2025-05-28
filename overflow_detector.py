import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.utils import resize
import io
import time

class OverflowDetector:
    def __init__(self, model_path='models/overflow_detector_v1_with_backbone.pt'):
        """
        Initialize the overflow detector with MobileNetV3 model.
        
        Args:
            model_path: Path to the trained model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['off', 'safe', 'unsafe']
        
        # Initialize model
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(in_features=1024, out_features=len(self.classes))
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.classes}")
    
    def apply_jpeg_compression(self, image, quality=75):
        """
        Apply JPEG compression to match training data preprocessing.
        
        Args:
            image: PIL Image
            quality: JPEG quality (75 matches extract_frames.py default)
            
        Returns:
            PIL Image with JPEG compression applied
        """
        # Save to memory buffer as JPEG and reload
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        # Convert to RGB if needed (JPEG might change mode)
        if compressed_image.mode != 'RGB':
            compressed_image = compressed_image.convert('RGB')
        return compressed_image
    
    def predict(self, image, apply_compression=True):
        """
        Run inference on a PIL image.
        
        Args:
            image: PIL Image
            apply_compression: Whether to apply JPEG compression to match training data
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities, timing_info)
        """
        start_time = time.time()
        
        # Preprocessing timing
        preprocess_start = time.time()
        
        # Resize image first (matches training preprocessing)
        image = resize(image)
        
        # Apply JPEG compression to match training data
        if apply_compression:
            image = self.apply_jpeg_compression(image)
        
        # Apply transforms and run inference
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        preprocess_time = time.time() - preprocess_start
        
        # Model inference timing
        inference_start = time.time()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            predicted_class = self.classes[preds.item()]
            confidence = probabilities[0][preds.item()].item()
            all_probs = probabilities[0].cpu().numpy()
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Create timing info dictionary
        timing_info = {
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'total_time': total_time,
            'fps': 1.0 / total_time if total_time > 0 else 0
        }
        
        return predicted_class, confidence, all_probs, timing_info
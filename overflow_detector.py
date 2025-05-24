import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from utils.utils import resize

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
            transforms.Resize((224, 224)),  # Ensure consistent size after resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.classes}")
    
    def predict(self, image, already_transformed=False):
        """
        Run inference on a PIL image.
        
        Args:
            image: PIL Image
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        if not already_transformed:
            image = resize(image)
            image = self.transform(image)
        
        input_tensor = image.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            predicted_class = self.classes[preds.item()]
            confidence = probabilities[0][preds.item()].item()
            all_probs = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, all_probs
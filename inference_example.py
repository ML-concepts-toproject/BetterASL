"""
ASL Model Inference Script for Raspberry Pi
Usage: python inference.py <image_path>
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import sys
import os

# Load model info
with open("exported_models/model_info.json", "r") as f:
    model_info = json.load(f)

IMG_SIZE = model_info["img_size"]
CLASS_NAMES = model_info["class_names"]
MEAN = model_info["mean"]
STD = model_info["std"]

# Define image preprocessing (same as validation)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def load_model(model_path="exported_models/asl_model_half.pt"):
    """Load the TorchScript model"""
    device = "cpu"  # Raspberry Pi typically uses CPU
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, model_path

def predict(model, image_path, model_path=None):
    """Run inference on a single image"""
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Convert to half-precision if needed (for half-precision models)
    if model_path and "half" in model_path.lower():
        img_tensor = img_tensor.half()
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0].float(), dim=0)  # Convert to float for softmax
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    return predicted_class, confidence_pct, probabilities

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Loading model...")
    model, model_path = load_model()
    
    print(f"Running inference on {image_path}...")
    predicted_class, confidence, probs = predict(model, image_path, model_path=model_path)
    
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Show top 3 predictions
    top3_probs, top3_indices = torch.topk(probs, 3)
    print(f"\nTop 3 predictions:")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
        print(f"  {i+1}. {CLASS_NAMES[idx.item()]:8s} ({prob.item()*100:5.2f}%)")

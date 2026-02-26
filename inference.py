import argparse
import torch
from torchvision import transforms
from PIL import Image

from model import get_model

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = get_model(num_classes=4, pretrained=False)
    try:
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        print(f"Loaded weights from {args.model_weights}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model = model.to(device)
    model.eval()
    
    # Load and transform Image
    transform = get_inference_transforms()
    try:
        image = Image.open(args.image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # Interpret Attributes
    print(f"Attributes for {args.image_path}:")
    
    # Handle single element batch vs sequence
    if probabilities.ndim == 0:
        probabilities = [probabilities]
        
    for i, prob in enumerate(probabilities):
        is_present = 1 if prob > args.threshold else 0
        print(f"Attr{i+1}: {is_present} (prob: {prob:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('image_path', type=str, help='Path to test image')
    parser.add_argument('--model_weights', type=str, default='model_weights.pth', help='Path to trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for positive classification')
    
    args = parser.parse_args()
    predict(args)

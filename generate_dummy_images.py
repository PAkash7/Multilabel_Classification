import os
from PIL import Image

def generate_dummy_images(label_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(label_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    for line in lines:
        if line.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(output_dir, line)
            if not os.path.exists(img_path):
                # Create a simple 224x224 black image
                img = Image.new('RGB', (224, 224), color = (73, 109, 137))
                img.save(img_path)
                
if __name__ == '__main__':
    generate_dummy_images('labels.txt', 'images')
    print("Generated dummy images.")

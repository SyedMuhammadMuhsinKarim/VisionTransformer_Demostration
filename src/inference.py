import torch
from model import ViTObjectDetection
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Configuration
MODEL_PATH = "checkpoints/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8  # Number of classes

def load_model():
    model = ViTObjectDetection(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def detect_objects(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_logits, bbox_coords = model(image_tensor)

    return class_logits, bbox_coords

def display_detection(image_path, class_logits, bbox_coords):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    # Add bounding boxes and labels to the image
    # (Implement logic to visualize bbox_coords on the image)
    plt.show()

if __name__ == "__main__":
    model = load_model()
    img_path = "data/images/test/example.jpg"
    class_logits, bbox_coords = detect_objects(img_path, model)
    display_detection(img_path, class_logits, bbox_coords)

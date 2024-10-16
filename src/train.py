import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import YOLODataset
from model import ViTObjectDetection
from utils import calculate_loss, calculate_metrics

# Configuration settings
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, val_loader, num_classes):
    # Initialize model, optimizer, and loss function
    model = ViTObjectDetection(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            labels = targets["labels"].to(DEVICE)
            boxes = targets["boxes"].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            class_logits, bbox_coords = model(images)

            # Compute loss
            loss = calculate_loss(class_logits, bbox_coords, labels, boxes)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader)}")

        # Validation after each epoch
        validate_model(model, val_loader)

    # Save the trained model
    torch.save(model.state_dict(), "checkpoints/best_model.pth")


def validate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            labels = targets["labels"].to(DEVICE)
            boxes = targets["boxes"].to(DEVICE)

            # Forward pass
            class_logits, bbox_coords = model(images)

            # Compute metrics
            calculate_metrics(class_logits, bbox_coords, labels, boxes)

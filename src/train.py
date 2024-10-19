import torch
import torch.optim as optim
from src.model import ViTObjectDetection
from src.utils import calculate_loss, calculate_metrics

# Configuration settings
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            batch_labels = []
            batch_boxes = []

            for target in targets:
                if target is not None:
                    batch_labels.append(target[:, 0])  # assuming the first column is the class label
                    batch_boxes.append(target[:, 1:])   # the remaining columns are bounding boxes

            max_length = max(len(lbl) for lbl in batch_labels) if batch_labels else 0
            num_images = len(batch_labels)

            labels = torch.zeros((num_images, max_length), dtype=torch.long).to(DEVICE)
            boxes = torch.zeros((num_images, max_length, 4), dtype=torch.float).to(DEVICE)

            for idx, (lbl, bxs) in enumerate(zip(batch_labels, batch_boxes)):
                labels[idx, :len(lbl)] = lbl
                boxes[idx, :len(bxs)] = bxs

            # Forward pass
            class_logits, bbox_coords = model(images)

            # Compute metrics
            calculate_metrics(class_logits, bbox_coords, labels, boxes)


def train_model(train_loader, val_loader, num_classes):
    """ Train the ViT Object Detection model 
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_classes (int): Number of classes in the dataset    
    """
    model = ViTObjectDetection(num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(DEVICE)

            batch_labels = []
            batch_boxes = []

            for target in targets:
                if target is not None:
                    batch_labels.append(target[:, 0])  # assuming the first column is the class label
                    batch_boxes.append(target[:, 1:])   # the remaining columns are bounding boxes

            max_length = max(len(lbl) for lbl in batch_labels) if batch_labels else 0
            num_images = len(batch_labels)

            labels = torch.zeros((num_images, max_length), dtype=torch.long).to(DEVICE)
            boxes = torch.zeros((num_images, max_length, 4), dtype=torch.float).to(DEVICE)

            for idx, (lbl, bxs) in enumerate(zip(batch_labels, batch_boxes)):
                labels[idx, :len(lbl)] = lbl
                boxes[idx, :len(bxs)] = bxs

            optimizer.zero_grad()
            class_logits, bbox_coords = model(images)

            cls_labels = labels[:, 0]

            loss = calculate_loss(class_logits, bbox_coords, cls_labels, boxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(train_loader)}")

        validate_model(model, val_loader)
        

    torch.save(model.state_dict(), "checkpoints/best_model.pth")

def custom_collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        if sample is not None:  # In case some images have no labels
            img, target = sample
            images.append(img)
            targets.append(target)
    
    images = torch.stack(images)
    
    return images, targets

# if __name__ == "__main__":
#     train_dataset = YOLODataset("data/images/train", "data/labels/train", target_size=(224, 224))
#     val_dataset = YOLODataset("data/images/val", "data/labels/val", target_size=(224, 224))

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

#     train_model(train_loader, val_loader, num_classes=8)


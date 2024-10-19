# from regex import D
# from torch.utils.data import DataLoader
# import torch

# from src.train import custom_collate_fn
# from src.dataset import YOLODataset


# """ Test the YOLODataset class
# Shapes: 
#     Images shape: torch.Size([16, 3, 512, 512]), 
#     Targets shape: 16 (Batch size)
#     Target shape: torch.Size([x, 5]) where x is object count in the image and 5 is class_id, x1, y1, x2, y2
# """

BATCH_SIZE = 16
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# n = 1
# width = 512
# height = 512
# target_size = (width, height) # Target size for resizing images
# dataset = YOLODataset("data/images/train", "data/labels/train", target_size=target_size)
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# for images, targets in train_loader: # Load a batch of images and labels
#     print(type(images), type(targets))
#     print(len(images), len(targets))
#     print(f"Images shape: {images.shape}, Targets shape: {len(targets)}")
    
#     batch_labels = []
#     batch_boxes = []

#     for target in targets:
#         if target is not None:
#             batch_labels.append(target[:, 0])  # assuming the first column is the class label
#             batch_boxes.append(target[:, 1:])   # the remaining columns are bounding boxes

    
#     max_length = max(len(lbl) for lbl in batch_labels) if batch_labels else 0
#     num_images = len(batch_labels)

#     labels = torch.zeros((num_images, max_length), dtype=torch.long).to(DEVICE)
#     boxes = torch.zeros((num_images, max_length, 4), dtype=torch.float).to(DEVICE)

#     for idx, (lbl, bxs) in enumerate(zip(batch_labels, batch_boxes)):
#         labels[idx, :len(lbl)] = lbl
#         boxes[idx, :len(bxs)] = bxs

#     print(f"Batch Shapes: Labels: {labels.shape}, Boxes: {boxes.shape}")
    
    
    
#     break

from src.train import custom_collate_fn, train_model
from src.dataset import YOLODataset
from torch.utils.data import DataLoader

train_dataset = YOLODataset("data/images/train", "data/labels/train", target_size=(224, 224))
val_dataset = YOLODataset("data/images/val", "data/labels/val", target_size=(224, 224))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

train_model(train_loader, val_loader, num_classes=8)



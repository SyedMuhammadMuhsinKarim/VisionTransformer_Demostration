import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

def load_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            label = [float(x) for x in line.strip().split()]
            
            class_id, x_center, y_center, width, height = label
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            labels.append([class_id, x1, y1, x2, y2])
    
    return np.array(labels)

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = load_yolo_labels(label_path)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            image = transformed["image"]
            labels = transformed["bboxes"]
        
        return image, torch.tensor(labels)

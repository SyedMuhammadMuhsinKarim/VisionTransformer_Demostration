from hashlib import sha1
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
        """ Initialize the YOLO dataset
        
        Args:
            image_dir (str): Path to the directory containing images
            label_dir (str): Path to the directory containing YOLO labels
            transforms (albumentations.Compose): Image transforms to apply
            
        Raises:
            ValueError: If no images are found in the image directory
            FileNotFoundError: If the label directory does not exist
        """
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_filenames = os.listdir(image_dir)
         
        if len(self.image_filenames) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        if not os.path.exists(label_dir):
            raise ValueError(f"Label directory {label_dir} does not exist")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Get image and labels for a given index
        Args:
            idx (int): Index of the image to retrieve
            
        Returns:
            image (np.array): Image array
            labels (torch.Tensor): Bounding box labels
        """
        img_filename = self.image_filenames[idx]
        
        img_path = os.path.join(self.image_dir, img_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img_base_name, img_extension = os.path.splitext(img_filename)

        label_path = os.path.realpath(__file__)
        split_path = label_path.split("/")
        project_dir = "/".join(split_path[:-2])
        
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        label_path = os.path.join(project_dir, self.label_dir, img_base_name + ".txt")  
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        image = cv2.imread(img_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be opened: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        labels = load_yolo_labels(label_path)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            image = transformed["image"]
            labels = transformed["bboxes"]
        
        return image, torch.tensor(labels)

# if __name__ == "__main__":
    # dataset = YOLODataset("data/images/train", "data/labels/train")
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     image, labels = dataset[i]
    #     print(image.shape, labels.shape)

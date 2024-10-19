import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

def load_yolo_labels(file_path, img_w, img_h):
    """ Load YOLO labels from a file
    
    Args:
        file_path (str): Path to the YOLO label file
        img_w (int): Image width
        img_h (int): Image height
        
    Returns:
        np.array: Bounding box labels
    """
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            label = [float(x) for x in line.strip().split()]
            
            if len(label) == 5: 
                class_id, x_center, y_center, width, height = label
                
                x_center *= img_w
                y_center *= img_h
                width *= img_w
                height *= img_h
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                labels.append([class_id, x1, y1, x2, y2])
            else:
                print(f"Invalid label: {label}")
    
    return np.array(labels) if len(labels) > 0 else np.zeros((0, 5))  # Ensure labels have shape (N, 5)

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None, target_size=(512, 512)):
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
        self.target_size = target_size  # Add target size for resizing
        self.image_filenames = os.listdir(image_dir)
         
        self.filter_empty_labels()
                    
        if len(self.image_filenames) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        if not os.path.exists(label_dir):
            raise ValueError(f"Label directory {label_dir} does not exist")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")
        
    def filter_empty_labels(self):
        """ Filter out images with empty labels """
        valid_images = []
        for img_filename in self.image_filenames:
            img_base_name, _ = os.path.splitext(img_filename)
            label_path = os.path.join(self.label_dir, f"{img_base_name}.txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    if len(f.readlines()) > 0:
                        valid_images.append(img_filename)
            else:
                print(f"Label file not found: {label_path}")
        
        self.image_filenames = valid_images
        

    def __len__(self):
        return len(self.image_filenames)

    def __resize_image_label(self, image, labels):
        """ Resize image and labels to the target size 
        
        Args:
            image (np.array): Image array
            labels (np.array): Bounding box labels
            
        Returns:
            image (np.array): Resized image array
            labels (np.array): Resized bounding box labels
        """
        
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, self.target_size) # Resize the image to 512 x 512 x 3
        
        labels[:, 1:] *= np.array([self.target_size[0] / orig_w, 
                                self.target_size[1] / orig_h,  
                                self.target_size[0] / orig_w,  
                                self.target_size[1] / orig_h])
        return image, labels    
    
    def __getitem__(self, idx):
        """ Retrieve an image and its labels from the dataset
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
        
        img_base_name, _ = os.path.splitext(img_filename)
        label_path = os.path.join(self.label_dir, f"{img_base_name}.txt")
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or could not be opened: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        labels = load_yolo_labels(label_path, orig_w, orig_h)  # Load with original image size
        
        if labels.shape[0] == 0:
            return None
        
        image, labels = self.__resize_image_label(image, labels)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            image = transformed["image"]
            labels = transformed["bboxes"]
        
        image = torch.tensor(image).permute(2, 0, 1)
        
        labels = torch.tensor(labels)
        
        return image, labels
    
    def search_index(self, filename):
        """ Search for the index of an image in the dataset """
        for i, img_filename in enumerate(self.image_filenames):
            if img_filename == filename:
                return i
        
        raise ValueError(f"Image {filename} not found in the dataset")            

def custom_collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        if sample is not None:  # In case some images have no labels
            img, target = sample
            images.append(img)
            targets.append(target)  # Append the target as it is (do not convert to tensor here)
    
    images = torch.stack(images)
    
    return images, targets


if __name__ == "__main__":
    n = 1
    width = 512
    height = 512
    target_size = (width*n, height*n)
    dataset = YOLODataset("data/images/train", "data/labels/train", target_size=target_size)
    from torch.utils.data import DataLoader

    BATCH_SIZE = 16
    # collate_fn = lambda x: list(filter(lambda y: y is not None, x))

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    
    for images, targets in train_loader:
        print(type(images), type(targets))
        print(len(images), len(targets))
        print(f"Images shape: {images.shape}, Targets shape: {len(targets)}")
        print(f"Targets: {targets}")
        break
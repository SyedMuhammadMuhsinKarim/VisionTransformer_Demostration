import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn as nn
import torch
import os

matplotlib.use('Agg')

# def calculate_loss(class_logits, bbox_coords, labels, boxes):
#     classification_loss = nn.CrossEntropyLoss()
#     regression_loss = nn.MSELoss()
    
#     print(f"Class logits shape: {class_logits.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Bbox coords shape: {bbox_coords.shape}")
#     print(f"Boxes shape: {boxes.shape}")

#     cls_loss = classification_loss(class_logits, labels)

#     reg_loss = regression_loss(bbox_coords, boxes)

#     total_loss = cls_loss + reg_loss
#     return total_loss

def calculate_loss(class_logits, bbox_coords, labels, boxes):
    """ Calculate the loss for the model 
    
    Args:
        class_logits: Predicted class logits
        bbox_coords: Predicted bounding box coordinates
        labels: Ground truth labels
        boxes: Ground truth bounding boxes
        
    Returns:
        total_loss: Total loss for the model
    """
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.MSELoss()
    
    if labels.ndim > 1:
        labels = labels[:, 0]  # Adjust according to your label handling logic

    labels = labels.to(class_logits.device)

    cls_loss = classification_loss(class_logits, labels)

    boxes_selected = boxes[:, 0, :]  # Get the first box for each sample

    if bbox_coords.shape != boxes_selected.shape:
        raise ValueError(f"Bounding box predictions and targets must have the same shape: "
                         f"got {bbox_coords.shape} and {boxes_selected.shape}")

    boxes_selected = boxes_selected.to(bbox_coords.device)

    reg_loss = regression_loss(bbox_coords, boxes_selected)

    total_loss = cls_loss + reg_loss
    return total_loss

def calculate_metrics(class_logits, bbox_coords, labels, boxes):
    """ Calculate the metrics for the model """
    
    predicted_classes = torch.argmax(class_logits, dim=1)
    accuracy = (predicted_classes == labels).float().mean().item()
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

class ShowImage:
    def __init__(self, figsize=(10, 10)):
        self.figsize = figsize

    def __call__(self, image, labels):
        fig, ax = plt.subplots(1, figsize=self.figsize)
        ax.imshow(image)
        print(labels)
        for box in labels:
            label, x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{int(label)}", color='red', fontsize=12)
        
        plt.show()

class SaveImage:
    """ Save the image with bounding boxes """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def __call__(self, image, labels, filename):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        for box in labels:
            label, x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{int(label)}", color='red', fontsize=12)
        
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

class CutOutLabelsFromImage(ShowImage):
    """ Cut out the labels from the image """
    def __init__(self, figsize=(10, 10)):
        super().__init__(figsize)
    
    def __call__(self, image, labels):
        super().__call__(image, labels)
        for box in labels:
            label, x1, y1, x2, y2 = box
            cutout = image[int(y1):int(y2), int(x1):int(x2)] * 255
            print(f"Cutout shape: {cutout.shape}") # (0, 0, 3) if the cutout is empty
            if cutout.shape[0] == 0 or cutout.shape[1] == 0:
                print("Cutout is empty")
                continue
            
            print(f"Cutout shape: {cutout.shape}")
            
            plt.imshow(cutout)
            plt.show()

class PrepareSheet:
    """ Prepare a CSV sheet for the dataset 
    
    Args:
        dataset: Dataset object
        csv_path: Path to save the CSV
    
    Returns:
    """
    def __init__(self, dataset, csv_path):
        self.dataset = dataset
        self.csv_path = csv_path
        
    def __call__(self):
        """ Prepare the CSV sheet """
        
        with open(self.csv_path, 'w') as f:
            f.write("Image Path,Class ID,X1,Y1,X2,Y2\n")
            for i in range(len(self.dataset)):
                result = self.dataset[i]
                if result is not None:
                    image_path = self.dataset.image_filenames[i]
                    image, labels = result
                    for label in labels:
                        class_id, x1, y1, x2, y2 = label
                        f.write(f"{image_path},{class_id},{x1},{y1},{x2},{y2}\n")
                else:
                    continue

import torch
import torch.nn as nn
from transformers import ViTModel

class ViTObjectDetection(nn.Module):
    """ Vision Transformer (ViT) Object Detection model """
    def __init__(self, num_classes, pretrained=True):
        """ Initialize the ViT Object Detection model
        
        Args:
            num_classes (int): Number of classes in the dataset
            pretrained (bool): Load pre-trained weights or not
            
        raises:
            ValueError: If the number of classes is less than 1
        """
        super(ViTObjectDetection, self).__init__()
        
        # Load Vision Transformer (ViT) model from Hugging Face Transformers
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") if pretrained else ViTModel()

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.regressor = nn.Linear(self.vit.config.hidden_size, 4) 
         
    def forward(self, images):
        """ Forward pass of the ViT Object Detection model
        Model Configuration:
            - ViTModel: Vision Transformer model
            - classifier: Linear layer for classification
            - regressor: Linear layer for bounding box regression
            
        Args:
            images (torch.Tensor): Input image tensor

        Returns:
            class_logits (torch.Tensor): Predicted class logits
            bbox_coords (torch.Tensor): Predicted bounding box coordinates
        """
        vit_outputs = self.vit(pixel_values=images).last_hidden_state
        
        cls_token = vit_outputs[:, 0]

        class_logits = self.classifier(cls_token)
        bbox_coords = self.regressor(cls_token)

        return class_logits, bbox_coords

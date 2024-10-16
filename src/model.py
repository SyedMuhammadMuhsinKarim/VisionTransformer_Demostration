import torch
import torch.nn as nn
from transformers import ViTModel

class ViTObjectDetection(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTObjectDetection, self).__init__()
        # Load Vision Transformer (ViT) model from Hugging Face Transformers
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k") if pretrained else ViTModel()

        # Detection head
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.regressor = nn.Linear(self.vit.config.hidden_size, 4)  # For bounding box regression

    def forward(self, images):
        # Pass through Vision Transformer (ViT)
        vit_outputs = self.vit(pixel_values=images).last_hidden_state
        
        # Take the CLS token's embedding (used for classification)
        cls_token = vit_outputs[:, 0]

        # Predict class and bounding boxes
        class_logits = self.classifier(cls_token)
        bbox_coords = self.regressor(cls_token)

        return class_logits, bbox_coords

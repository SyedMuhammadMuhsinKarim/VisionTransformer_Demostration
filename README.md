# Vision Transformer (ViT) for Object Detection with YOLO Labels

This project implements an object detection system using the Vision Transformer (ViT) model. The dataset is structured using YOLO format annotations, and the model is trained using PyTorch.

## Table of Contents

- [Vision Transformer (ViT) for Object Detection with YOLO Labels](#vision-transformer-vit-for-object-detection-with-yolo-labels)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Directory Structure](#directory-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
  - [Dataset Preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
  - [License](#license)

## Project Overview

This project leverages the Vision Transformer (ViT) for object detection tasks using YOLO-formatted datasets. We use the transformer architecture for image feature extraction and add a custom detection head for bounding box regression and class prediction.

## Directory Structure

The folder structure of the project is as follows:

```bash
project_root/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── src/
│   ├── dataset.py          # Dataset loading and preprocessing
│   ├── model.py            # ViT model with detection head
│   ├── train.py            # Training logic and validation
│   ├── inference.py        # Inference logic for testing new images
│   └── utils.py            # Helper functions (loss, metrics)
│
├── main.py         
├── checkpoints/            # Saved model weights
└── README.md               # Project documentation
```

## Installation

To set up this project, clone the repository and install the required dependencies.

### Prerequisites

Ensure that you have `Python 3.8+` installed, along with `pip`.

### Clone the Repository

```bash
git clone https://github.com/SyedMuhammadMuhsinKarim/VisionTransformer_Demostration
cd VisionTransformer_Demostration
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the dataset from [this link](https://drive.google.com/file/d/1OS3qGTbYhzoH0PA6PrnkTt1vbq5B1x09/view?usp=sharing).
2. Extract the dataset into the `data/images/` directory.
3. Ensure the labels are in the `data/labels/` directory, following the YOLO format.

## Training the Model

To train the model, navigate to the main directory and run:

```bash
python main.py
```

## Acknowledgements

- [PyTorch](https://pytorch.org/) - Deep learning framework used for model training.
- [Vision Transformers](https://arxiv.org/abs/2010.11929) - The architecture leveraged for object detection.

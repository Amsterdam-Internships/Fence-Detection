import os
import torch

# problem details
CLASSES = 1

# model details
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'

# data details
TRAIN_IMAGE_PATH = os.path.join('..', '..', 'data', 'images')
VALID_IMAGE_PATH = TRAIN_IMAGE_PATH

TRAIN_ANNOTATIONS_PATH = os.path.join('..', '..', 'data', 'fences-quays', 'annotations', 'train-annotations-6px.json')
VALID_ANNOTATIONS_PATH = os.path.join('..', '..', 'data', 'fences-quays', 'annotations', 'valid-annotations-6px.json')

# training details
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRECISION = 'mixed' # mixed or single precision floats

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 1

NUM_WORKERS = 3
NUM_EPOCHS = 50

LR = 8e-5
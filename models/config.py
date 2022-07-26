import os
import torch

TITLE = 'resnet18-unet-1600s-aug-11px-blobs'

# problem details
CLASSNAME = 'blobs' #'fence'
CLASSES = 1

BLOBS = True # polygon-like annotations

# model details
ENCODER_DETAILS = 'resnet18'
ENCODER_WEIGHTS = 'imagenet' # 'imagenet'
ACTIVATION = 'sigmoid'

# data details
TRAIN_IMAGE_PATH = os.path.join('..', 'data', 'fences-quays', 'images')
VALID_IMAGE_PATH = TRAIN_IMAGE_PATH

TRAIN_ANNOTATIONS_PATH = os.path.join('..', 'data', 'fences-quays', 'annotations', 'train-annotations-11px.json')
VALID_ANNOTATIONS_PATH = os.path.join('..', 'data', 'fences-quays', 'annotations', 'valid-annotations-11px.json')

if BLOBS:
    TRAIN_ANNOTATIONS_PATH = os.path.join('..', 'data', 'polygon-fences')
    VALID_ANNOTATIONS_PATH = TRAIN_ANNOTATIONS_PATH

LOGS_PATH = os.path.join('..', 'experiments')

# training details
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRECISION = 'mixed' # mixed or single precision floats

PREPROCESSING = False
AUGMENTATION = True

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

NUM_WORKERS = 3
NUM_EPOCHS = 30

LR = 8e-5
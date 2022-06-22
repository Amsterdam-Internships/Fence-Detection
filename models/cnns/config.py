import os
import torch

TITLE = 'effnetb3-unetpp-1000s-aug'

# problem details
CLASSNAME = 'fence'
CLASSES = 1

# model details
ENCODER_DETAILS = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet' # 'imagenet'
ACTIVATION = 'sigmoid'

# data details
TRAIN_IMAGE_PATH = os.path.join('..', '..', 'data', 'fences-quays', 'images')
VALID_IMAGE_PATH = TRAIN_IMAGE_PATH

TRAIN_ANNOTATIONS_PATH = os.path.join('..', '..', 'data', 'fences-quays', 'annotations', 'train-annotations-11px.json')
VALID_ANNOTATIONS_PATH = os.path.join('..', '..', 'data', 'fences-quays', 'annotations', 'valid-annotations-11px.json')

LOGS_PATH = os.path.join('..', '..', 'experiments')

# training details
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRECISION = 'mixed' # mixed or single precision floats

PREPROCESSING = False
AUGMENTATION = True

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4

NUM_WORKERS = 3
NUM_EPOCHS = 30

LR = 8e-5
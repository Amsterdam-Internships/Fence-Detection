import os


ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet' #os.path.join('..', 'weights', 'resnet50-19c8e357.pth')
CLASSES = 1
ACTIVATION = 'sigmoid'

TRAIN_IMAGE_PATH = os.path.join("..", "..", "..", "data", "images")
TEST_IMAGE_PATH = os.path.join("..", "..", "..", "data", "self-annotated-sample", "images")

TRAIN_ANNOTATIONS_PATH = os.path.join("..", "..", "..", "data", "spectrum-batch-1", "annotations.json")
TEST_ANNOTATIONS_PATH = os.path.join("..", "..", "..", "data", "self-annotated-sample", "annotations.json")
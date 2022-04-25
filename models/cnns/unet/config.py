# import the necessary packages
import torch
import os

# paths
TRAIN_IMAGE_PATH = os.path.join("..", "..", "..", "data", "images")
TEST_IMAGE_PATH = os.path.join("..", "..", "..", "data", "self-annotated-sample", "images")

TRAIN_ANNOTATIONS_PATH = os.path.join("..", "..", "..", "data", "spectrum-batch-1", "annotations.json")
TEST_ANNOTATIONS_PATH = os.path.join("..", "..", "..", "data", "self-annotated-sample", "annotations.json")

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 2
NUM_LEVELS = 2

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 5e-4
NUM_EPOCHS = 40
BATCH_SIZE = 16

# define the input image dimensions
INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_HEIGHT = 512

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join("..", "..", "weights", "unet.pth")
PLOT_PATH = os.path.join("..", "..", "..", "experiments", "unet")
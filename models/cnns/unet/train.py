import os
import sys
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm


import config
from model import UNet

sys.path.insert(0, os.path.join('..', '..', '..'))
from loaders.datasets import AmsterdamDataset


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # print(inputs)
        # print(targets)
        # assert False
        # inputs = inputs > 0.5
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


if __name__ == '__main__':
    # define transformations
    transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT,
        config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()])

    # create the train and test datasets
    train = AmsterdamDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_ANNOTATIONS_PATH, transform=transform)
    test = AmsterdamDataset(config.TEST_IMAGE_PATH, config.TEST_ANNOTATIONS_PATH, transform=transform)

    print(f"[INFO] found {len(train)} examples in the training set...")
    print(f"[INFO] found {len(test)} examples in the test set...")

    # create the training and test data loaders
    train_loader = DataLoader(train, shuffle=True,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=3)

    test_loader = DataLoader(test, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=3)

    # initialize our UNet model
    unet = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    # lossFunc = BCEWithLogitsLoss(pos_weight=torch.as_tensor([1000]).to(config.DEVICE))
    # lossFunc = BCELoss().to(config.DEVICE)
    lossFunc = IoULoss().to(config.DEVICE)
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # calculate steps per epoch for training and test set
    trainSteps = len(train) // config.BATCH_SIZE
    testSteps = len(test) // config.BATCH_SIZE

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    print("[INFO] training the network...")
    startTime = time.time()

    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set

        for (i, (x, y)) in enumerate(train_loader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in test_loader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)

    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)
import os
import sys
import torch
import config

from torchvision import transforms
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join('..', '..', '..'))
from loaders.datasets import AmsterdamDataset

if __name__ == '__main__':
    # define metric
    iou = JaccardIndex(num_classes=2).to(config.DEVICE)

    # define transform
    transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT,
        config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()])

    # create the train and test datasets
    train = AmsterdamDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_ANNOTATIONS_PATH, transform=transform)
    test = AmsterdamDataset(config.TEST_IMAGE_PATH, config.TEST_ANNOTATIONS_PATH, transform=transform)

    # create the training and test data loaders
    train_loader = DataLoader(train, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=3)

    test_loader = DataLoader(test, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=3)

    # load model
    model = torch.load(config.MODEL_PATH).to(config.DEVICE)
    model.eval()

    # report IoU on test set
    for x, y in test_loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE).int()
        pred = model(x)
        pred = torch.sigmoid(pred)

        # print(pred > .5)
        print(torch.min(pred))
        print(torch.max(pred))
        print(torch.count_nonzero((pred > .5).to(torch.int32)))
        
        print(pred.shape)
        print(y.shape)
        print(iou(pred, y))

    # # report IoU on train set
    # for x, y in train_loader:
    #     x, y = x.to(config.DEVICE), y.to(config.DEVICE).int()
    #     pred = model(x)
    #     pred = torch.sigmoid(pred)
    #     print(iou(pred, y))
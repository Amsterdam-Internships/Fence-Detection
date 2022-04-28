import os
import sys
import torch
import config

from model import UNet
from torchvision import transforms
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import albumentations as album
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join('..', '..', '..'))
from loaders.datasets import AmsterdamDataset


def to_tensor(x, **kwargs):
    # print(x.shape)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform  W  
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)


class IoULoss(nn.Module):
    __name__ = 'iou_loss'

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class BCEWithLogitsLoss(nn.Module):
    __name__ = 'bce_with_logits_loss'

    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50]).cuda())
        return loss(inputs, targets)

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    __name__ = 'focal_loss'

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


class PositiveIoUScore(nn.Module):
    __name__ = 'iou_score'

    def __init__(self):
        super(PositiveIoUScore, self).__init__()

    def forward(self, inputs, targets):
        ious = JaccardIndex(num_classes=2, reduction='none')(inputs.cpu(), targets.int().cpu())
        return ious[1]

class NegativeIoUScore(nn.Module):
    __name__ = 'bg_iou'

    def __init__(self):
        super(NegativeIoUScore, self).__init__()

    def forward(self, inputs, targets):
        ious = JaccardIndex(num_classes=2, reduction='none')(inputs.cpu(), targets.int().cpu())
        return ious[0]



if __name__ == '__main__':
    model = UNet
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)

    train_transform = transforms.Compose([transforms.ToPILImage(),
    # transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    #     config.INPUT_IMAGE_WIDTH)),
    # transforms.RandomHorizontalFlip(p=.5),
    # transforms.RandomRotation(degrees=(-25, 25)),
    # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor()])

    valid_transform = transforms.Compose([transforms.ToPILImage(),
    # transforms.Resize((config.INPUT_IMAGE_HEIGHT,
    #     config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()])

    train_dataset = AmsterdamDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_ANNOTATIONS_PATH,
                                    # transform=train_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = AmsterdamDataset(config.TEST_IMAGE_PATH, config.TEST_ANNOTATIONS_PATH, 
                                    # transform=valid_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn))

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=3)

    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

    # Set num of epochs
    EPOCHS = 10

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define loss function
    # loss = smp.utils.losses.BCEDiceLoss()
    loss = BCEWithLogitsLoss()

    # define metrics
    metrics = [
        PositiveIoUScore(),
        NegativeIoUScore(),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.00008),
    ])

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth'):
        model = torch.load('../input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth', map_location=DEVICE)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:

        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')
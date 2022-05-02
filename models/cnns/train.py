import os
import sys
import torch
import config

from model import FPN, UNet, UNetPP, PSPNet
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join('..', '..'))
from loaders.datasets import AmsterdamDataset
from utils.augmentation import *
from utils.metrics import *
from utils.train import TrainEpoch, ValidEpoch


if __name__ == '__main__':
    # get decoder
    model = UNet

    # get encoding and training augmentation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS)
    train_transform = get_amsterdam_augmentation()

    train_dataset = AmsterdamDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_ANNOTATIONS_PATH,
                                    transform=train_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = AmsterdamDataset(config.VALID_IMAGE_PATH, config.VALID_ANNOTATIONS_PATH, 
                                    preprocessing=get_preprocessing(preprocessing_fn))

    # get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        PositiveIoUScore(),
        NegativeIoUScore(),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=config.LR),
    ])

    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=config.DEVICE,
        precision=config.PRECISION,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=config.DEVICE,
        precision=config.PRECISION,
        verbose=True,
    )

    best_iou_score = 0.
    train_logs_list, valid_logs_list = [], []

    # training loop
    for i in range(0, config.NUM_EPOCHS):

        # perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
import sys
import torch
import config

from model import *
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

sys.path.insert(0, '..')
from loaders.datasets import AmsterdamDataset
from utils.augmentation import *
from utils.metrics import *
from utils.train import TrainEpoch, ValidEpoch
from utils.log import TrainLog


if __name__ == '__main__':
    # dedicated log
    train_logs = TrainLog(config.TITLE, dirpath=config.LOGS_PATH)

    # get decoder
    model = UNet

    train_logs.add_model_data(model)
    train_logs.add_config_data(config)


    # get encoding and training augmentation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config.ENCODER, config.ENCODER_WEIGHTS) \
                       if config.PREPROCESSING else None

    train_transform = get_amsterdam_augmentation() if config.AUGMENTATION else None

    train_dataset = AmsterdamDataset(config.TRAIN_IMAGE_PATH, config.TRAIN_ANNOTATIONS_PATH,
                                    transform=train_transform,
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                    classname=config.CLASSNAME,
                                    train=False)
    valid_dataset = AmsterdamDataset(config.VALID_IMAGE_PATH, config.VALID_ANNOTATIONS_PATH, 
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                    classname=config.CLASSNAME,
                                    train=False)

    # get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        PositiveIoUScore(),
        NegativeIoUScore(),
        TrueNegativeRate(),
        TruePositiveRate(),
        FalseNegativeRate(),
        FalsePositiveRate(),
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
        train_results = train_epoch.run(train_loader, save=False)
        valid_results = valid_epoch.run(valid_loader, save=True)

        if config.CLASSNAME == 'fence':
            biou_valid = BlobOverlap()
            biou_valid.update(valid_epoch.predictions, valid_epoch.targets)

        train_logs.add_metrics(name='train',
                               epoch=i,
                               dice_loss=train_results['dice_loss'],
                               positive_iou=train_results['iou_score'],
                               negative_iou=train_results['bg_iou'],
                               true_negative_rate=train_results['tnr'],
                               false_positive_rate=train_results['fpr'],
                               false_negative_rate=train_results['fnr'],
                               true_positive_rate=train_results['tpr'])

        train_logs.add_metrics(name='valid',
                               epoch=i,
                               dice_loss=valid_results['dice_loss'],
                               positive_iou=valid_results['iou_score'],
                               negative_iou=valid_results['bg_iou'],
                               true_negative_rate=valid_results['tnr'],
                               false_positive_rate=valid_results['fpr'],
                               false_negative_rate=valid_results['fnr'],
                               true_positive_rate=valid_results['tpr'],
                               blob_iou=biou_valid.compute() if config.CLASSNAME == 'fences' else 0)

        # save model if a better val IoU score is obtained
        if best_iou_score < valid_results['iou_score']:
            best_iou_score = valid_results['iou_score']
            torch.save(model, os.path.join(config.LOGS_PATH, config.TITLE, 'best_model.pth'))
            print('Model saved!')
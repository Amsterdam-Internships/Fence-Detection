import cv2

import numpy as np
import albumentations as A

from functools import partial


def to_tensor(x, **kwargs):
    """ transform image output to tensor format
    """
    return x.transpose(2, 0, 1).astype('float32')


def to_rolled(x, **kwargs):
    """ roll image along horizontal axis
    """
    return np.roll(x, kwargs['roll'], axis=1)


def get_preprocessing(preprocessing_fn=None):
    """ construct preprocessing encoder transform 
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)


def RandomHorizontalRoll(**kwargs):
    """ roll image random amount of pixels along horizontal axis
    """
    roll = np.random.randint(0, kwargs['max_roll'])
    f = partial(to_rolled, roll=roll)
    
    return A.Lambda(image=f, mask=f, p=kwargs['p'])


def get_amsterdam_augmentation(height=512, width=1024, p=.5, border_mode=cv2.BORDER_REPLICATE):
    """ return segmentation augmentation for amsterdam dataset
    """
    transform = A.Compose([
                A.HorizontalFlip(p=p),
                RandomHorizontalRoll(max_roll=int(width / 2), p=p),
                A.Perspective(scale=(.1, .1), keep_size=True, p=p),
                A.Rotate(limit=2.5, p=p, border_mode=cv2.BORDER_REPLICATE),
                A.RandomSizedCrop(min_max_height=(height - 50, height), height=height, width=width, p=p),
                A.RandomBrightnessContrast(p=p),
                A.GaussNoise(p=p),
                # A.CoarseDropout(max_holes=2, max_height=int(height / 2), max_width=int(width / 2), mask_fill_value=0, p=p),
            ])

    return transform

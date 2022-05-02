import cv2

import albumentations as A


def to_tensor(x, **kwargs):
    """ transform image output to tensor format
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """ construct preprocessing encoder transform 
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
    _transform.append(A.Lambda(image=to_tensor, mask=to_tensor))
        
    return A.Compose(_transform)


def get_amsterdam_augmentation(height=512, width=1024, p=.5, border_mode=cv2.BORDER_REPLICATE):
    """ return segmentation augmentation for amsterdam dataset
    """
    transform = A.Compose([
                A.HorizontalFlip(p=p),
                A.Rotate(limit=2.5, p=p, border_mode=cv2.BORDER_REPLICATE),
                A.RandomSizedCrop(min_max_height=(height - 50, height), height=height, width=width, p=p),
                A.RandomBrightnessContrast(p=p),
                A.GaussNoise(p=p),
            ])

    return transform

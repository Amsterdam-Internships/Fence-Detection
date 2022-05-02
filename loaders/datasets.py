import os
import io
import json
import torch

import numpy as np

from PIL import Image

from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import mask as cmask
from pycocotools.coco import COCO


class COCODataset(Dataset):
    """
    """
    def __init__(self, root_dir, subset='train', challenge='instances', task='segmentation', year=2017, transform=None, categories=''):
        """
        """
        self.task = task
        self.transform = transform

        # find directories
        self.image_dir = os.path.join(root_dir, f'{subset}{year}')
        self.ann_dir = f'annotations_trainval{year}'

        # find specific annotation directory
        for dirname in os.listdir(root_dir):
            if challenge in dirname:
                self.ann_dir = dirname
                break
        
        self.coco = COCO(os.path.join(root_dir, 
                                      self.ann_dir, 
                                      'annotations', 
                                      f'{challenge}_{subset}{year}.json'))

        # get annotations using COCO API
        category_ids = self.coco.getCatIds(catNms=categories)
        annotation_ids = self.coco.getAnnIds(catIds=category_ids)
        self.annotations = self.coco.loadAnns(annotation_ids)


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        """
        """
        annotation = self.annotations[idx]

        fname = f'{annotation["image_id"]}'.zfill(12) + '.jpg'
        
        image = io.imread(os.path.join(self.image_dir, fname))
        label = self.coco.annToMask(annotation) \
            if self.task == 'segmentation' else annotation[self.task]

        if self.transform:
            image = self.transform(image)
            label = torch.as_tensor(label)
            
        return image, label


class ADE20KDataset(Dataset):
    """
    """
    def __init__(self):
        return


    def __len__(self):
        return


    def __getitem__(self):
        return


class AmsterdamDataset(Dataset):
    """
    """
    def __init__(self, imagedir, annotations, transform=None, preprocessing=None, train=True):
        """
        """
        self.transform = transform
        self.preprocessing = preprocessing
        self.imagedir = imagedir
        
        # get annotations using COCO API
        self.coco = COCO(annotations)

        self.category_ids = self.coco.getCatIds()
        annotation_ids = self.coco.getAnnIds(catIds=self.category_ids)
        image_ids = self.coco.getImgIds(catIds=self.category_ids)

        self.images = self.coco.loadImgs(image_ids)
        self.annotations = self.coco.loadAnns(annotation_ids)

        self.fnames = [image['file_name'] for image in self.images]

        # filter only on fence annotations
        if train:
            tmp = []
            for image in self.images:
                ann_ids = self.coco.getAnnIds(imgIds=image['id'], catIds=self.category_ids, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                # check if image annotations contain fence(s)
                for ann in anns:
                    # if ann.get('attributes').get('Class') == 'Quay':
                    if ann.get('counts'):
                        tmp.append(image)
                        break

            self.images = tmp


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        """
        """
        # load image
        obj = self.images[idx]
        fname = os.path.join(self.imagedir, obj['file_name'])
        image = io.imread(fname)
        
        # get all annotations corresponding to image
        annotation_ids = self.coco.getAnnIds(imgIds=obj['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)

        # generate mask
        mask = np.zeros((obj['height'], obj['width']))

        for annotation in annotations:
            if annotation.get('counts'):
                # pass
                # decode uncompressed RLE
                ann = cmask.frPyObjects(annotation.get('counts'), obj['height'], obj['width'])
                ann = cmask.decode(ann)
                # mask = np.maximum(mask, np.array(ann) * annotation['category_id'])
                mask = np.maximum(mask, np.array(ann) * 1)
            else:
                pass
                # mask = np.maximum(mask, self.coco.annToMask(annotation) * annotation['category_id'])
                # mask = np.maximum(mask, self.coco.annToMask(annotation) * 1)

        # convert from float to integer
        mask = np.expand_dims(mask.astype(np.uint8), axis=-1)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
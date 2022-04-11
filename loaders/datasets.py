import os
import io
import json
import torch

from PIL import Image

from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
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
    def __init__(self, csv_file, root_dir, transform=None):
        """
        """
        self.metadata = pd.read_csv(csv_file, sep=', ', engine='python')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = os.path.join(self.root_dir, 
                             self.metadata.filename.iloc[idx])
        
        img = io.imread(fname)

        # TODO
        ann = None
        lab = np.zeros(3)
        lab[self.metadata.quay.iloc[idx]] = 1
        
        sample = [img, lab]

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample
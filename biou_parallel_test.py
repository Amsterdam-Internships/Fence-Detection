import os
import sys
import time
import torch

from torch.utils.data import DataLoader

# sys.path.insert(0, os.path.join('..', '..'))
from loaders.datasets import AmsterdamDataset
from utils.augmentation import *
from utils.metrics import BlobOverlap


VALID_IMAGE_PATH = os.path.join('data', 'fences-quays', 'images')
VALID_ANNOTATIONS_PATH = os.path.join('data', 'fences-quays', 'annotations', 'test-annotations-6px.json')

BATCH_SIZE = 4
NUM_WORKERS = BATCH_SIZE


if __name__ == '__main__':
    preprocessing_fn = None
    train_transform = get_amsterdam_augmentation()

    input_a = AmsterdamDataset(VALID_IMAGE_PATH, VALID_ANNOTATIONS_PATH,
                                    preprocessing=get_preprocessing(preprocessing_fn),
                                    train=False)

    input_loader = DataLoader(input_a, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    BlobIoU_single = BlobOverlap(num_workers=NUM_WORKERS)
    # BlobIoU_parallel = BlobOverlap(num_workers=NUM_WORKERS)

    all_preds = []
    all_targets = []

    model = torch.load(os.path.join('models', 'cnns', 'best_model.pth'))

    start = time.time()

    for i, (images, targets) in enumerate(input_loader):
        with torch.no_grad():
            preds = model(images.cuda())

        BlobIoU_single.update(preds, targets)
        
        all_preds.append(preds)
        all_targets.append(targets)

        print(i)

    print('single:', BlobIoU_single.compute())
    print(f'Wall time*: {round(time.time() - start, 2)}s')

    # start = time.time()

    # BlobIoU_parallel.update_all(all_preds, all_targets)

    # print('parallel:', BlobIoU_parallel.compute())
    # print(f'Wall time: {round(time.time() - start, 2)}s')
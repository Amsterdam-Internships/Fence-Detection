import os
import sys
import torch

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm import tqdm
from shapely.geometry import Point, Polygon

# relative imports
sys.path.insert(0, '..')
from utils.general import visualize

# figsize
plt.rcParams["figure.figsize"] = (10, 5)

# paths
PATH_IMAGE_DIR = os.path.join('..', 'data', 'images')
PATH_META_FILE = os.path.join('..', 'data', '15000-water-images', 'metadata.csv')

PATH_MODEL_FENCE = os.path.join('..', 'experiments', 'fences', 'effnetb5-unetpp-1600s-aug', 'best_model.pth')
PATH_MODEL_QUAY = None

PATH_SAVE_FILE = os.path.join('..', 'data', 'images-masks')

# load fence and quay models
model_fence = torch.load(PATH_MODEL_FENCE)
model_quay = None

# debug limit
n = np.inf

# load datadump metadata
metadata = pd.read_csv(PATH_META_FILE)

# change column to match image names
f = lambda x: x.replace('-equirectangular-panorama_8000.jpg', '')
metadata.filename_dump = metadata.filename_dump.apply(f)

if __name__ == '__main__':
    # inference loop
    for i in tqdm(metadata.index):
        row = metadata.iloc[i]

        lng = row.lng
        lat = row.lat

        fname = row.filename_dump

        # get image
        new_entry = False
        per_image = {'height_l':np.nan, 'height_r':np.nan}

        for side in ['l', 'r']:
            try:
                name = f'{fname}-{side}.jpg'
                img = plt.imread(os.path.join(PATH_IMAGE_DIR, name))
                new_entry = True
            except:
                continue

            x = img.transpose(2, 0, 1).astype('float32')
            x = torch.as_tensor(x).unsqueeze(0).cuda()

            # predict
            with torch.no_grad():
                y = model_fence(x)

            # to np array
            y = y.squeeze().cpu().numpy() > .5
            
            f, (ax1, ax2) = plt.subplots(1, 2)
            
            ax1.imshow(img)
            ax2.imshow(y)
            
            ax1.axis('off')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(PATH_SAVE_FILE, f'{lng}-{lat}-{side}.png'), dpi=96)
            plt.close(f)
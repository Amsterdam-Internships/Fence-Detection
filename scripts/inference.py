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


# paths
PATH_IMAGE_DIR = os.path.join('..', 'data', 'images')
PATH_META_FILE = os.path.join('..', 'data', '15000-water-images', 'metadata.csv')

PATH_MODEL_FENCE = os.path.join('..', 'experiments', 'resnet18-unet-1600s-aug-11px', 'best_model.pth')
PATH_MODEL_QUAY = None

PATH_SAVE_FILE = os.path.join('..', 'data', 'geometry', 'predictions.geojson')


if __name__ == '__main__':
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

    results = {'fname':[], 'timestamp':[], 'height_l':[], 'height_r':[], 'geometry':[]}

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
            
            # visualize(x=img, y=y)
            coords = np.asarray(np.where(y == 1)).T
            ys, xs = coords[:, 0], coords[:, 1]
            
            # calculate width coverage
            width = len(np.unique(xs)) / y.shape[1]
            
            # calculate pixel height
            height = 0
            
            if width > 0:
                samples = np.random.choice(xs, 20)
                for j, sample in enumerate(samples):
                    ys_per_x = coords[xs == sample, 0]
                    height_per_x = abs(ys_per_x.max() - ys_per_x.min())
                    height += height_per_x
                
                height /= (j + 1)
            
            per_image[f'height_{side}'] = height
            
        # save results
        if new_entry:
            results['fname'].append(f'{fname}-equirectangular-panorama_8000.jpg')
            results['timestamp'].append(row.timestamp)
            results['height_l'].append(per_image['height_l'])
            results['height_r'].append(per_image['height_r'])
            results['geometry'].append(Point(lng, lat))
        
        if i == n:
            break

    gdf = gpd.GeoDataFrame(results)
    gdf.to_file(PATH_SAVE_FILE, driver='GeoJSON') 
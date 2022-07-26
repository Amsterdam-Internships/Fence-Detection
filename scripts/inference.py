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

PATH_MODEL_FENCE = os.path.join('..', 'experiments', 'fences', 'effnetb6-unetpp-1600s-aug', 'best_model.pth')
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
            height = 0

            if len(coords) > 0:
                sorted_by_min_y = coords[coords[:, 0].argsort()[::-1]]
                max_y = sorted_by_min_y[0][0]
                
                sub_ys = sorted_by_min_y[:, 0]
                sub_xs = sorted_by_min_y[:, 1]
                
                y_range = sub_ys[sub_ys > (max_y - 50)]
                x_range = sub_xs[sub_ys > (max_y - 50)]
                
                min_y = y_range.min()
                min_x = x_range.min()
                max_x = x_range.max()
                
                # print('min_y', min_y)
                # print('max_y', max_y)
                
                # print('min_x', min_x)
                # print('max_x', max_x)
                # print('\n')
                sample_range = np.arange(min_x, max_x)

                if len(sample_range) > 20:
                    samples = np.random.choice(sample_range, 20)
                
                    for j, sample in enumerate(samples):
                        ys_per_x = coords[xs == sample, 0]
                        
                        if len(ys_per_x) >= 1:
                            height_per_x = abs(ys_per_x.min() - max_y)
                            height += height_per_x
                        else:
                            height += 1

                    height /= (j + 1)
            
                    # print('est. height', height)
            
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
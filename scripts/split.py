import os
import sys
import cv2
import csv

import matplotlib.pyplot as plt

sys.path.insert(0, '..')
from src.processing.loaders import PanoramaLoader
from src.processing.geometry import viewpoint_to_pixels


# panoramas load and save directories
LOAD = os.path.join('..', 'src', 'data', '15000-water-images')
SAVE = os.path.join('..', 'src', 'data', 'images')

# subsampling details
WIDTH = 1024
HEIGHT = 512
HORIZON = 2000
VIEWPOINT_OFFSET = 90

# limit for testing
LIMIT = None


def bbox_on_center(panorama, center, width=WIDTH, height=HEIGHT):
    center_y, center_x = center
    
    x_min, x_max = int(center_x - width / 2), int(center_x + width / 2)
    y_min, y_max = int(center_y - height / 2), int(center_y + height / 2)
    
    return slice(y_min, y_max), slice(x_min, x_max)


def write_to_csv(row, fname):
    with open(fname, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


if __name__ == '__main__':
    PL = PanoramaLoader(LOAD)
    PL.set_option('read_method', cv2.imread)
    PL.set_option('show_method', plt.imshow)

    meta_dump = os.path.join(SAVE, 'metadata.csv')
    open(meta_dump, 'w').close()
    headerrow = ['filename', 'bbox'] + PL[0]._metadata.index.tolist()

    write_to_csv(headerrow, meta_dump)

    for i, panorama in enumerate(PL):
        print('\r', end='')

        name = panorama.filename_dump.replace('-equirectangular-panorama_8000.jpg', '')

        left_center = panorama.viewpoint_back + VIEWPOINT_OFFSET
        right_center = panorama.viewpoint_front + VIEWPOINT_OFFSET

        left_slices = bbox_on_center(panorama, (HORIZON, viewpoint_to_pixels(left_center)))
        right_slices = bbox_on_center(panorama, (HORIZON, viewpoint_to_pixels(right_center)))

        left = panorama[left_slices]
        right = panorama[right_slices]

        left.save(os.path.join(SAVE, 'images', f'{name}-l.jpg'))
        right.save(os.path.join(SAVE, 'images', f'{name}-r.jpg'))

        left_row = [f'{name}_l', str([left_slices[0].stop, left_slices[1].start, WIDTH, HEIGHT])] + panorama._metadata.values.tolist()
        right_row = [f'{name}_r', str([right_slices[0].stop, right_slices[1].start, WIDTH, HEIGHT])] + panorama._metadata.values.tolist()

        write_to_csv(left_row, meta_dump)
        write_to_csv(right_row, meta_dump)

        if LIMIT:
            print(f'{i}/{LIMIT}', end='')
        else:
            print(f'{i}/{len(PL)}', end='')

        if LIMIT and i == LIMIT:
            break
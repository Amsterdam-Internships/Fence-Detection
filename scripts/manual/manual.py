import os
import sys
import json
import time
import shutil
import tkinter
import keyboard
import matplotlib
import subprocess

import numpy as np

sys.path.insert(0, '..')
from loaders.datasets import AmsterdamDataset


IMAGES = os.path.join('..', 'data', 'images')
SAVE_DIR = os.path.join('..', 'data', 'new-batch')


if __name__ == '__main__':
    # get file names
    fnames = os.listdir(IMAGES)
    annotations = os.path.join('..', 'data', 'spectrum-batch-1', 'annotations.json')

    with open(annotations) as f:
        used = json.load(f)
    
    used_fnames = [obj['file_name'] for obj in used['images']]
    fnames = [fname for fname in fnames if fname not in used_fnames]
    
    np.random.shuffle(fnames)

    for fname in fnames:
        fpath = os.path.join(IMAGES, fname)

        p = subprocess.Popen(['python', 'display.py', fpath])

        time.sleep(.1)

        listening = True
        while listening:
            if keyboard.is_pressed('right arrow'):
                p.kill()
                shutil.copy(fpath, os.path.join(SAVE_DIR, fname))
                listening = False
                print(len(os.listdir(SAVE_DIR)))
                break
            elif keyboard.is_pressed('left arrow'):
                p.kill()
                listening = False
                break
            elif keyboard.is_pressed('esc'):
                p.kill()
                listening = False
                sys.exit()
                break
            
            time.sleep(.1)
import os
import shutil

import numpy as np

FROM = os.path.join('..', 'data', 'new-batch')
NUMBER = 1000
TO = os.path.join('..', 'data', 'new-spectrum-batch')

if __name__ == '__main__':
    fnames = os.listdir(FROM)
    np.random.shuffle(fnames)

    for i, fname in enumerate(fnames):
        shutil.copy(os.path.join(FROM, fname), os.path.join(TO, fname))

        if i == (NUMBER - 1):
            break

    print(len(os.listdir(TO)))
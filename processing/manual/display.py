import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str)
    args = parser.parse_args()

    img = mpimg.imread(args.fname) #path to IMG
    plt.imshow(img)
    plt.show()
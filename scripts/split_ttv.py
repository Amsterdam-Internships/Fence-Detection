import os
import json

import numpy as np

from pycocotools.coco import COCO


READ_IMAGE_DIR = os.path.join('..', 'data', 'images')
READ_ANNOTATION_DIR = os.path.join('..', 'data', 'fences-quays', 'annotations')

WRITE_IMAGE_DIR = os.path.join('..', 'data', 'fences-quays', 'images')
WRITE_ANNOTATION_DIR = READ_ANNOTATION_DIR


def make_empty_coco_json(coco):
    """
    """
    coco_json = coco.dataset

    coco_json['images'] = []
    coco_json['annotations'] = []

    return coco_json


def make_coco_json(coco, fpath, imgs, anns):
    """
    """
    skeleton = make_empty_coco_json(coco)

    skeleton['images'] = imgs
    skeleton['annotations'] = anns

    with open(fpath, 'w') as f:
        json.dump(skeleton, f)

    return


def get_imgs_anns(coco, idxs, imgs):
    """
    """
    new_imgs, new_anns = [], []

    for i, idx in enumerate(idxs):
        img = imgs[idx]
        img_id = img['id']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)

        new_imgs += [img]
        new_anns += ann
    
    return new_imgs, new_anns


def get_coco_json_data(coco, fpath):
    """
    """
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)

    imgs = coco.loadImgs(img_ids)
    anns = coco.loadAnns(ann_ids)

    return imgs, anns


if __name__ == '__main__':

    # get coco annotations
    fname = 'train-annotations-2-6px.json'
    fpath = os.path.join(READ_ANNOTATION_DIR, fname)

    # use coco api
    coco = COCO(fpath)

    # get images and annotations
    all_imgs, _ = get_coco_json_data(coco, fpath)

    # create random valid and test subsets
    n = len(all_imgs)

    np.random.seed(1)
    idxs = np.random.permutation(n)

    idxs_valid = idxs[:100]
    idxs_test = idxs[100:200]
    idxs_train = idxs[200:]

    # create coco json for train subset
    imgs, anns = get_imgs_anns(coco, idxs_train, all_imgs)

    fpath = os.path.join(WRITE_ANNOTATION_DIR, 'train-annotations-6px.json')
    make_coco_json(coco, fpath, imgs, anns)

    # create coco json for validation subset
    imgs, anns = get_imgs_anns(coco, idxs_valid, all_imgs)

    fpath = os.path.join(WRITE_ANNOTATION_DIR, 'valid-annotations-6px.json')
    make_coco_json(coco, fpath, imgs, anns)

    # create coco json for test subset
    imgs, anns = get_imgs_anns(coco, idxs_test, all_imgs)

    fpath = os.path.join(WRITE_ANNOTATION_DIR, 'test-annotations-6px.json')
    make_coco_json(coco, fpath, imgs, anns)
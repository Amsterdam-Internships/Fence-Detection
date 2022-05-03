import os
import json
import shutil

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

    # copy images
    for img in imgs:
        if img['file_name'] not in os.listdir(WRITE_IMAGE_DIR):
            source_path = os.path.join(READ_IMAGE_DIR, img['file_name'])
            target_path = os.path.join(WRITE_IMAGE_DIR, img['file_name'])

            shutil.copy(source_path, target_path)

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


def get_filtered_anns(coco, filter_fn):
    """
    """
    anns = []

    for ann in coco.dataset['annotations']:
        if filter_fn(ann):
            anns.append(ann)

    ann_ids = [ann['id'] for ann in anns]
    img_ids = list(set([ann['image_id'] for ann in anns]))

    anns = coco.loadAnns(ann_ids)
    imgs = coco.loadImgs(img_ids)

    return imgs, anns


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
    # get coco annotations (first spectrum batch)
    fname = 'train-annotations-1-6px.json'
    fpath = os.path.join(READ_ANNOTATION_DIR, fname)

    # use coco api
    coco = COCO(fpath)

    fn = lambda x: not (not x.get('counts'))
    imgs, anns = get_filtered_anns(coco, filter_fn=fn)

    # get coco annotations (second spectrum batch)
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
    imgs_2, anns_2 = get_imgs_anns(coco, idxs_train, all_imgs)

    print(len(imgs), len(anns))
    print(len(imgs_2), len(anns_2))
    
    # combine first and second spectrum batch
    imgs += imgs_2
    anns += anns_2

    print(len(imgs), len(anns))

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
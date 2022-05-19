import os
import json
import shutil

import numpy as np

from pycocotools.coco import COCO


READ_IMAGE_DIR = os.path.join('..', 'data', 'images')
READ_ANNOTATION_DIR = os.path.join('..', 'data', 'fences-quays', 'annotations', 'batch-json')

WRITE_IMAGE_DIR = os.path.join('..', 'data', 'fences-quays', 'images')
WRITE_ANNOTATION_DIR = os.path.join('..', 'data', 'fences-quays', 'annotations')


def make_empty_coco_json(coco):
    """"""
    coco_json = coco.dataset

    coco_json['images'] = []
    coco_json['annotations'] = []

    return coco_json


def make_coco_json(coco, fpath, imgs, anns):
    """"""
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


def get_imgs_anns(coco, idxs, imgs, image_id_offset=0, annotation_id_offset=0):
    """"""
    new_imgs, new_anns = [], []

    for i, idx in enumerate(idxs):
        img = imgs[idx]
        img_id = img['id']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)

        new_imgs += [img]
        new_anns += ann
    
    return new_imgs, new_anns


def get_filtered_imgs_anns(coco, filter_fn):
    """"""
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
    """"""
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)

    imgs = coco.loadImgs(img_ids)
    anns = coco.loadAnns(ann_ids)

    return imgs, anns


def reset_ids(imgs, anns, imgs_offset=0, anns_offset=0):
    """"""
    # copy 
    imgs_copy, anns_copy = imgs.copy(), anns.copy()

    new_imgs, new_anns = [], []
    j = 0

    for i, img in enumerate(imgs_copy):
        for ann in anns_copy:
            # find all corresponding annotations
            if ann['image_id'] == img['id']:
                # reset annotation id and reference to img
                ann = ann.copy()
                ann['id'] = j + anns_offset
                ann['image_id'] = i + imgs_offset
                new_anns += [ann]
                
                # increment annotation id
                j += 1

        # reset image id
        img['id'] = i + imgs_offset
        new_imgs += [img]

    return new_imgs, new_anns


if __name__ == '__main__':
    # get coco annotations (first spectrum batch)
    fname = 'annotations-6px-batch-1.json'
    fpath = os.path.join(READ_ANNOTATION_DIR, fname)

    # use coco api
    coco = COCO(fpath)

    # fn = lambda x: not (not x.get('counts'))
    fn = lambda x: True
    imgs, anns = get_filtered_imgs_anns(coco, filter_fn=fn)

    print(len(imgs))

    # get coco annotations (second spectrum batch)
    fname = 'annotations-6px-batch-2.json'
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
    imgs_2, anns_2 = reset_ids(imgs_2, anns_2, imgs_offset=len(imgs), 
                                               anns_offset=len(anns))
    
    # combine first and second spectrum batch
    imgs += imgs_2
    anns += anns_2

    # fpath = os.path.join(WRITE_ANNOTATION_DIR, 'train-annotations-6px.json')
    # make_coco_json(coco, fpath, imgs, anns)

    # # create coco json for validation subset
    # imgs, anns = get_imgs_anns(coco, idxs_valid, all_imgs)
    # imgs, anns = reset_ids(imgs, anns)

    # fpath = os.path.join(WRITE_ANNOTATION_DIR, 'valid-annotations-6px.json')
    # make_coco_json(coco, fpath, imgs, anns)

    # # create coco json for test subset
    # imgs, anns = get_imgs_anns(coco, idxs_test, all_imgs)
    # imgs, anns = reset_ids(imgs, anns)

    # fpath = os.path.join(WRITE_ANNOTATION_DIR, 'test-annotations-6px.json')
    # make_coco_json(coco, fpath, imgs, anns)
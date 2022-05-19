import os
import json
import shutil

import numpy as np
import pandas as pd

from pycocotools.coco import COCO


READ_METADATA_DIR = os.path.join('..', 'data', 'fences-quays')

READ_ANNOTATION_DIR = os.path.join('..', 'data', 'fences-quays', 'annotations', 'batch-json')
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


def get_imgs_anns_by_fnames(images, annotations, fnames):
    """"""
    imgs, anns = [], []

    # get images
    for image in all_images:
        if image['file_name'] in fnames:
            imgs.append(image)

    img_ids = [img['id'] for img in imgs]

    # get annotations
    for annotation in all_annotations:
        if annotation['image_id'] in img_ids:
            anns.append(annotation)

    return imgs, anns


def read_json(fpath):
    """"""
    with open(fpath, 'r') as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    # set seed for predictable randomness
    np.random.seed(1)

    metadata = pd.read_csv(os.path.join(READ_METADATA_DIR, 'metadata.csv'))

    batch_a = read_json(os.path.join(READ_ANNOTATION_DIR, 'annotations-1-6px.json'))
    batch_b = read_json(os.path.join(READ_ANNOTATION_DIR, 'annotations-2-6px.json'))

    coco = COCO(os.path.join(READ_ANNOTATION_DIR, 'annotations-1-6px.json'))

    images_a, annotations_a = reset_ids(batch_a['images'], batch_a['annotations'])
    images_b, annotations_b = reset_ids(batch_b['images'], batch_b['annotations'], imgs_offset=len(images_a),
                                                                                   anns_offset=len(annotations_a))

    all_images = images_a + images_b
    all_annotations = annotations_a + annotations_b

    tuples = list(metadata.Buurtcode.value_counts().to_dict().items())
    N = len(tuples)

    idxs = list(np.random.permutation(N))

    numbas = np.zeros(3)
    i = 0

    splits = {'0':[], '1':[], '2':[]}
    counts = np.zeros(3)

    while idxs:
        idx = idxs.pop()

        splits[str(i)].append(tuples[idx][0])
        counts[i] += tuples[idx][1]

        if counts[i] > 1680 and i == 0:
            i += 1
        elif counts[i] > 150 and i == 1:
            i += 1

    codes_train = splits['0']
    codes_valid = splits['1']
    codes_test = splits['2']

    metadata['subset'] = ''

    for subset in ['train', 'valid', 'test']:
        # get fnames of images within the subset
        df = metadata.query(f'Buurtcode in @codes_{subset}')
        fnames = df.filename_x.to_list()
        indices = df.index.to_list()
        metadata.subset.iloc[indices] = subset
        # get corresponding annotations
        imgs, anns = get_imgs_anns_by_fnames(all_images, all_annotations, fnames)
        # write to subset json
        make_coco_json(coco, os.path.join(WRITE_ANNOTATION_DIR, f'{subset}-annotations-6px.json'), imgs, anns)

    metadata.to_csv(os.path.join(READ_METADATA_DIR, 'metadata.csv'))

    
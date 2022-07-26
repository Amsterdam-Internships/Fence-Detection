import os
import sys
import cv2
import json
import argparse

import numpy as np

from tqdm import tqdm
from lxml import etree
from skimage import measure
from pycocotools import mask
from itertools import groupby

sys.path.insert(0, '..')
from utils.metrics import to_blobs

# width of of polyline polygon-like mask
PIXELS = 11

# polygon-like polyline annotations
BLOBS = True 


def binary_mask_to_rle(binary_mask):
    """"""
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def polyline_to_polygon(points, width, height, pixels=PIXELS):
    """"""
    points = points.astype(int)
    polygon = []

    # create ndarray binary mask
    background = np.zeros((height, width)).astype(int)
    binary_mask = cv2.polylines(background, [points], False, 1, pixels, lineType=cv2.LINE_AA)

    if BLOBS:
        binary_mask = to_blobs(binary_mask, eps=50)[0]

    # RLE encode
    arr = np.asfortranarray(binary_mask, dtype=np.uint8)
    counts = binary_mask_to_rle(arr)

    # compress
    rle = mask.frPyObjects(counts, height, width)

    return counts, rle


def parse_polygon(index, image, polygon):
    """"""
    json_object = {}

    width = int(image.attrib.get('width'))
    height = int(image.attrib.get('height'))

    # convert annotation and image id
    json_object['id'] = index
    json_object['image_id'] = int(image.attrib.get('id'))
    json_object['category_id'] = 1

    # convert polygon points to ndarray
    points = polygon.attrib.get('points').split(';')
    points = [point.split(',') for point in points]
    points = np.array([np.array(points, dtype=float).flatten()])

    json_object['segmentation'] = points.tolist()
    
    # encode
    rles = mask.frPyObjects(points, height, width)
    rle = mask.merge(rles)

    json_object['area'] = mask.area(rle).astype(float)
    json_object['bbox'] = mask.toBbox(rle).tolist()

    # additional attributes
    json_object['iscrowd'] = 0
    json_object['attributes'] = {attribute.attrib.get('name'):attribute.text \
        for attribute in polygon.getchildren()}

    return [json_object]


def parse_polyline(index, image, polyline):
    """"""
    json_object = {}

    width = int(image.attrib.get('width'))
    height = int(image.attrib.get('height'))

    # convert annotation and image id
    json_object['id'] = index
    json_object['image_id'] = int(image.attrib.get('id'))
    json_object['category_id'] = 2

    # convert multiline points to ndarray
    points = polyline.attrib.get('points').split(';')
    points = [point.split(',') for point in points]
    points = np.array(points, dtype=float)

    # convert ndarray to polygon RLE
    counts, rle = polyline_to_polygon(points, width, height)
    
    json_object['counts'] = counts
    json_object['area'] = mask.area(rle).astype(float)
    json_object['bbox'] = mask.toBbox(rle).tolist()

    # additional attributes
    json_object['iscrowd'] = 0
    json_object['attributes'] = {attribute.attrib.get('name'):attribute.text \
        for attribute in polyline.getchildren()}

    return [json_object]


def parse_image(index, image):
    """"""
    annotations = []

    # convert image
    img = [{
        'id': int(image.attrib.get('id')),
        'width': int(image.attrib.get('width')),
        'height': int(image.attrib.get('height')),
        'file_name': image.attrib.get('name'),
        'flickr_url': '',
        'coco_url': '',
        'date_captured': 0,
    }]

    # convert annotations
    for elem in image:
        if elem.tag == 'polygon':
            annotations += parse_polygon(index, image, elem)
            index += 1

        elif elem.tag == 'polyline':
            annotations += parse_polyline(index, image, elem)
            index += 1
    
    return index, img, annotations


def parse_meta(meta):
    """"""
    # TODO: update license
    licenses = [{'name': '', 'id': 0, 'url': ''}]

    # TODO: update info
    task = etree.SubElement(meta, 'task')
    info = {
        'contributor': 'Jorrit Ypenga',
        'date_created': '',
        'url': '',
        'version': '',
        'year': 2022,
    }

    labels = etree.SubElement(meta, 'labels')
    categories = []

    for i, label in enumerate(labels.getchildren()):
        categories += [{'id': i + 1, 
                        'name': etree.SubElement(label, name).text(), 
                        'supercategory': ''}]

    return licenses, info, categories


def xml_to_json(xml):
    """"""
    json_object = {}

    # meta
    json_object['licenses'] = None
    json_object['info'] = None
    json_object['categories'] = None

    # images and annotations
    json_object['images'] = []
    json_object['annotations'] = []

    index = 0
    for child in tqdm(xml):
        if child.tag == 'meta':
            licenses, info, categories = parse_meta(child)

            # set meta
            json_object['licenses'] = licenses
            json_object['info'] = info
            json_object['categories'] = categories

        elif child.tag == 'image':
            index, img, anns = parse_image(index, child)

            json_object['images'] += img
            json_object['annotations'] += anns
    
    return json_object


if __name__ == '__main__':
    for i in range(2):
        dirname = os.path.join('..', 'data', 'fences-quays', 'annotations', 'xml')
        fname = os.path.join(dirname, f'annotations-{i + 1}.xml')

        with open(fname) as f:
            xml = f.read().encode('ascii')

        xml = etree.fromstring(xml)
        json_object = xml_to_json(xml)

        # dump
        dirname = os.path.join('..', 'data', 'fences-quays', 'annotations', 'batch-json')
        fname = f'annotations-{i + 1}-{PIXELS}px' + '-blobs' if BLOBS else ''
        f = open(os.path.join(dirname, f'{fname}.json'), 'w')
        json.dump(json_object, f)
        f.close()
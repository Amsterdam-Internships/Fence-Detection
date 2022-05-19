import os
import shutil

from lxml import etree


READ_IMAGE_DIR = os.path.join('..', 'data', 'fences-quays', 'images')
READ_XML_DIR = os.path.join('..', 'data', 'fences-quays', 'annotations', 'xml')

WRITE_IMAGE_DIR = os.path.join('..', 'data', 'cvat_import_2', 'images')
WRITE_XML_DIR = os.path.join('..', 'data', 'cvat_import_2')


if __name__ == '__main__':
    fname = 'annotations-2.xml'
    fpath = os.path.join(READ_XML_DIR, fname)

    with open(fpath) as f:
        xml = f.read().encode('ascii')

    xml = etree.fromstring(xml)

    # create directories for export
    os.mkdir(WRITE_XML_DIR)
    os.mkdir(WRITE_IMAGE_DIR)

    # copy annotations
    shutil.copy(os.path.join(READ_XML_DIR, fname), os.path.join(WRITE_XML_DIR, 'annotations.xml'))

    # copy images
    for child in xml:
        if child.tag == 'image':
            imgname = child.attrib.get('name')
            shutil.copy(os.path.join(READ_IMAGE_DIR, imgname), os.path.join(WRITE_IMAGE_DIR, imgname))

    # zip
    # shutil.make_archive('cvat_import', 'zip', WRITE_XML_DIR)
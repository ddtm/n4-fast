from __future__ import print_function

import os
import sys
import struct
import numpy as np
import scipy.misc
import scipy.io
import cv2

def read_edges(filename):
    with open(filename, 'rb') as f:
        bytes = f.read(8)

        size = struct.unpack('II', bytes)

        edges = np.fromfile(f, dtype=np.single, count=(size[0] * size[1]))
        edges = edges.reshape(size)


    return edges

def combine_maps(maps, output_size):
    output = np.zeros(output_size, dtype=np.single)

    for t in maps:
        scale = 1.0 / t['scale']
        shift = t['shift']
        path = t['path']

        edges = read_edges(path)

        edges = cv2.resize(edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        add_border_y = max(output_size[0] - (edges.shape[0] + shift), 0)
        add_border_x = max(output_size[1] - (edges.shape[1] + shift), 0)
        edges = cv2.copyMakeBorder(
            edges, shift, add_border_y, shift, add_border_x, cv2.BORDER_REFLECT)

        edges = edges[0 : output_size[0], 0 : output_size[1]]

        output += edges

    output /= len(maps)

    return output

if __name__ == '__main__':
    images_base_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/BSDS500/BSR/BSDS500/data/images/test'
    images_list_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/BSDS500/BSR/BSDS500/data/bsds500_test.txt'

    sources = [
        './output',
        './output_3'
    ]

    target_path = './combined'

    maps_template = [
        {
            'scale': 2.0,
            'shift': 1
        },
        {
            'scale': 1.0,
            'shift': 2
        },
        {
            'scale': 0.5,
            'shift': 4
        }
    ]

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Prepare images lists.
    images_list = [os.path.join(images_base_path, line.strip()) for line in open(images_list_path)]

    for image_path in images_list:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img = scipy.misc.imread(image_path)

        maps = []
        for source in sources:
            for t in maps_template:
                map_fullname = os.path.join(
                    source, '%g' % t['scale'], '%s.bin' % image_name)

                maps += [dict({'path': map_fullname}.items() + t.items())]

        # print(maps)
        # sys.exit(0)

        edges = combine_maps(maps, img.shape[: 2])

        target_fullname = os.path.join(target_path, '%s.png' % image_name)
        scipy.misc.imsave(target_fullname, edges)

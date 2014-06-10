from __future__ import print_function

import os
import argparse
import struct
import subprocess
import multiprocessing as mp
import numpy as np
import scipy.misc
import scipy.io
import cv2

# DEVNULL = open(os.devnull, 'wb')
DEVNULL = None

def read_edges(filename):
    with open(filename, 'rb') as f:
        bytes = f.read(8)

        size = struct.unpack('II', bytes)

        edges = np.fromfile(f, dtype=np.single, count=(size[0] * size[1]))
        edges = edges.reshape(size)

    # print('Max', edges.max())
    # print('Min:', edges.min())

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

class Worker(mp.Process):
    def __init__(self, list_path, target_path, maps, device_id):
        super(Worker, self).__init__()

        self.list_path = list_path
        self.target_path = target_path
        self.maps = maps
        self.device_id = device_id

    def run(self):
        images_list = [line.strip() for line in open(self.list_path)]

        # Run N^4.
        for t in self.maps:
            maps_dir = os.path.join(self.target_path, '%g' % t['scale'])

            subprocess.call(
                'LD_LIBRARY_PATH=/usr/local/cuda-6.0/lib64 ../../bin/n4 '
                '-s {0} -t {1} --scale={2} -d {3}'.format(
                    self.list_path, maps_dir, t['scale'], self.device_id), 
                shell=True, stdout=DEVNULL)

        # # Combine edge maps.
        # for image_path in images_list:
        #     image_name = os.path.splitext(os.path.basename(image_path))[0]

        #     img = scipy.misc.imread(image_path)

        #     for t in maps:
        #         map_fullname = os.path.join(
        #             self.target_path, '%g' % t['scale'], '%s.bin' % image_name)

        #         t['path'] = map_fullname

        #     edges = combine_maps(self.maps, img.shape[: 2])

        #     target_fullname = os.path.join(self.target_path, '%s.png' % image_name)
        #     scipy.misc.imsave(target_fullname, edges)

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test stage of the N^4 algorithm.')
    parser.add_argument('-b', '--images-basepath', dest='images_base_path')
    parser.add_argument('-l', '--list-path', dest='images_list_path')
    parser.add_argument('-t', '--target-path', dest='target_path')

    args = parser.parse_args()

    print(args)

    if args.images_base_path:
        images_base_path = args.images_base_path
    else:
        images_base_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/Datasets/NYU/Source/data/images/test'

    if images_base_path == 'NYU':
        images_base_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/Datasets/NYU/Source/data/images/test'

    if args.images_list_path:
        images_list_path = args.images_list_path
    else:
        images_list_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/Datasets/NYU/Source/data/nyu_test.txt'

    if args.target_path:
        target_path = args.target_path
    else:
        target_path = './output'

    num_jobs = 2

    device_ids = [0, 1]

    maps = [
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

    # Make directories.
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for t in maps:
        maps_dir = os.path.join(target_path, '%g' % t['scale'])

        if not os.path.exists(maps_dir):
            os.makedirs(maps_dir)

    # Prepare images lists.
    images_list = [line.strip() for line in open(images_list_path)]
    # images_list = images_list[: 2]
    jobs_images_lists = [images_list[i :: num_jobs] for i in xrange(num_jobs)]

    jobs_images_lists_paths = []
    for i, l in enumerate(jobs_images_lists):
        jobs_images_lists_paths += ['chunk_%d.txt' % i]
        with open(jobs_images_lists_paths[-1], 'w') as f:
            for line in jobs_images_lists[i]:
                f.write(os.path.join(images_base_path, line) + '\n')

    jobs = []
    for i in range(num_jobs):
        p = Worker(jobs_images_lists_paths[i], target_path, maps, device_ids[i])
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()

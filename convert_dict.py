from __future__ import print_function

import os
import glob
import struct
import numpy as np
import scipy.io

dict_patches_path = '/home/yganin/Arbeit/Projects/NN/Segmentation/Lab/Reconstruction/samples_nyu.mat'

def load_samples(filename):
    if os.path.exists(filename):
        samples = scipy.io.loadmat(filename)
        samples = samples['samples']
        return samples

    if filename.endswith('.mat'):
        filename = filename[: -4]

    files = glob.glob('%s_*.mat' % filename)
    num_files = len(files)

    samples = []

    for i in xrange(num_files):
        s = scipy.io.loadmat('%s_%d.mat' % (filename, i))
        s = s['samples']
        
        samples += [s]

    samples = np.vstack(tuple(samples))

    return samples 

def process_samples():
    samples = load_samples(dict_patches_path)

    num_samples = len(samples)
    print(num_samples)

    with open('dict_processed_patches.bin', 'wb') as f:
        f.write(struct.pack('I', num_samples))

        for i in xrange(num_samples):
            sample = np.single(samples[i][0])

            if sample.max() > 1.0:
                sample = sample / 255.0
                print(sample.max())

            sample -= np.array([0.4911259, 0.42236657, 0.4007874, 0.21501289], dtype=np.single).reshape((1, 1, sample.shape[2]))

            if i == 0:
                print(sample.flags)
                print(sample.shape)

            sample.transpose((2, 0, 1)).flatten().tofile(f)

process_samples()

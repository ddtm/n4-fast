from __future__ import print_function

import os
import glob
import argparse
import struct
import numpy as np
import scipy.misc

def dump_matrix(m, filename):
	with open(filename, 'wb') as f:
	    f.write(struct.pack('II', m.shape[0], m.shape[1]))

	    m = np.single(m)
	    m.tofile(f)

parser = argparse.ArgumentParser(description='Converts depth into binary format.')
parser.add_argument('-p', '--path', required=True, dest='depths_path')

args = parser.parse_args()

depths_path = args.depths_path

files = glob.glob(os.path.join(depths_path, '*.png'))

for f in files:
	depth = np.single(scipy.misc.imread(f))
	depth = (depth - 713.0) / (9990.0 - 713.0)

	image_name = os.path.splitext(os.path.basename(f))[0]
	dump_matrix(depth, os.path.join('./depth', '%s.bin' % image_name))

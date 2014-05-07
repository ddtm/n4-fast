from __future__ import print_function

import os
import options
import cPickle
import struct
import numpy as np

def unpickle(filename):
    fo = open(filename, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

checkpoint_path = '/media/storage1/yganin/Arbeit/Checkpoints/pca+more_maps+add_fc+drop_3/ConvNet__2014-02-10_01.59.04_1/156000.2'

nnet = unpickle(checkpoint_path)

nnet = nnet['model_state']['layers']

guid = '\x6D\x6C\xFB\x72\x3A\x5C\x4C\x5E\x95\x66\x02\x9D\x2E\x64\x90\x45'

total_layers = sum([1 if l['type'] in ['conv', 'pool', 'fc', 'neuron'] else 0 for l in nnet])

print(total_layers)

with open('weights.bin', 'wb') as f:
	f.write(guid)
	f.write(struct.pack('I', total_layers))

	for l in nnet:
		if not 'weights' in l:
			if not l['type'] in ['conv', 'pool', 'fc', 'neuron']:
				continue

			f.write(struct.pack('I', 0))
		else:
			maps = l['weights'][0].shape[-1]

			if 'filterSize' in l:
				# We are dealing with a convolution.
				filter_size = l['filterSize'][0]
			else:
				# We are dealing with a FC layer.
				filter_size = 1

			weights = l['weights'][0].reshape((-1, filter_size, filter_size, maps))
			biases = l['biases'][0].flatten()

			print(l['name'])
			print(weights.shape)
			print(weights.size)
			# print(biases.shape)
			# print(weights.dtype)

			weights = weights.transpose((3, 0, 1, 2))
			# print(weights.flags)

			f.write(struct.pack('I', 2))

			f.write(struct.pack('I', weights.size))
			weights.tofile(f)
			
			f.write(struct.pack('I', biases.size))
			biases.tofile(f)

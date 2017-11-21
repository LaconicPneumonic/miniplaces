from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import cv2
import numpy as np
import scipy
import tensorflow as tf

from DataLoader import *
from imagenet_main import resnet_model_fn
from resnet_model import imagenet_resnet_v2

  
# Dataset Parameters
path_save = './build/resnet_model/model.ckpt-5'
model_dir = './build/resnet_model'
images_dir = '../../data/images/test/'
resnet_size = 50 
data_format = 100
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
classes = 100



x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
network = imagenet_resnet_v2(resnet_size, classes, None) 
logits = network(inputs=x, is_training=False)

# read data info from lists
 
def main():

	idx = 0

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(path_save + ".meta")
		save_path = tf.train.latest_checkpoint(model_dir)
		saver.restore(sess, save_path)
   
		testFileNames = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
		testFileNames.sort()
		testImagesLen = len(testFileNames)
 		
		with open('submit/submit.txt', 'w') as f:
			for i in range(testImagesLen):
				if i%100 == 0:
					print(i)

				images_batch = np.zeros((batch_size, fine_size, fine_size, c)) 
				images_name = []
	
				for i in range(batch_size):
					image = scipy.misc.imread(os.path.join(images_dir, testFileNames[idx]))
					image = scipy.misc.imresize(image, (load_size, load_size))
					image = image.astype(np.float32)/255. - data_mean
					offset_h = (load_size-fine_size)//2
					offset_w = (load_size-fine_size)//2

					images_batch[i, ...] = image[offset_h:offset_h+fine_size, offset_w:offset_w+fine_size, :]
					images_name.append(testFileNames[idx])							
					idx += 1

					if idx == testImagesLen:
						break

				feed_dict = {x: images_batch}   
				top_5 = sess.run(tf.nn.top_k(logits, k=5, sorted=True, name=None), feed_dict)
				
				for line, im_name in zip(top_5[1], images_name):
					f.write("test/" + im_name + " " + " ".join([str(line[i]) for i in range(5)])+ "\n")
	return 0
		 
if __name__ == '__main__':
  main()

# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image, ImageFile
import numpy as np
import shutil
import errno
import torch
import os
import random
from io import BytesIO


'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ChinadrinkDataset(data.Dataset):
	

	def __init__(self, root, mode='train', transform=None, target_transform=None, size = 28):
		'''
		The items are (filename,category). The index of all the categories can be found in self.idx_classes
		Args:
		- root: the directory where the dataset will be stored
		- transform: how to transform the input
		- target_transform: how to transform the target
		- download: need to download the dataset
		'''
		super(ChinadrinkDataset, self).__init__()
		self.root = root
		self.size = size
		self.transform = transform
		self.target_transform = target_transform

		if not self._check_exists():
			raise RuntimeError(
				'Dataset not found. ')

		self.classes, self.all_items = get_current_classes(os.path.join(self.root))

		#print(type(self.classes))
		#print(self.classes)

		self.idx_classes = index_classes(self.all_items)

		#print(self.all_items)
		#print(len(self.all_items))

		#paths, self.y = self.all_items, self.classes

		
		paths, self.y = zip(*[self.get_path_label(pl)
							  for pl in range(len(self))])
			
				

		self.x = map(load_img, paths, range(len(paths)))
		self.x = list(self.x)
		#print(self.x)

	def __getitem__(self, idx):
		x = self.x[idx]
		if self.transform:
			x = self.transform(x)
		#print(x, self.y[idx], idx)
		return x, self.y[idx]

	def __len__(self):
		return len(self.all_items)

	def get_path_label(self, index):
		
		img = self.all_items[index]
		target = int(os.path.basename(os.path.dirname(self.all_items[index])))
		if self.target_transform is not None:

			target = self.target_transform(target)

		return img, target

	def _check_exists(self):
		return os.path.exists(os.path.join(self.root))

def index_classes(items):
	idx = {}
	for i in items:
		if (not i in idx):
			#print('i',i)
			idx[i] = len(idx)
	print("== Dataset: Found %d images" % len(idx))
	return idx


def get_current_classes(fname):

	class_folders = [os.path.join(fname, SKU) for SKU in os.listdir(fname) \
					if os.path.isdir(os.path.join(fname, SKU))]

	#print(len(class_folders))
	
	classes = []
	
	for img_path in class_folders:
		
		#try:

		#Image.open(img_path)
		temp = [os.path.join(img_path, img_file) for img_file in os.listdir(img_path)]
		
		if len(temp) >= 10:

			classes.append(os.path.basename(img_path))

	#print(len(classes))
	image_folders = [os.path.join(fname, SKU, image) for SKU in classes for image in os.listdir(os.path.join(fname, SKU))]
	#print(len(image_folders))

	print("== Dataset: Found %d classes with 10 samples" %len(classes))
	return classes, image_folders


def load_img(path, idx):
	#print(len(path))
	#print(path)
		#for p in path:
	
	if path in IMG_CACHE:

		x = IMG_CACHE[path]
	else:

		rot = random.choice([0,90,180,270])

		'''with open(path, 'rb') as f:

			i = BytesIO()    
			bytimg = Image.open(f) 
			bytimg.save(i,"JPEG")
			i.seek(0)
			bytimg = i.read()
			dataBytesIO = BytesIO(bytimg)
		'''
		try:
			x = Image.open(path).convert("RGB")  #L for grayscale

			IMG_CACHE[path] = x
			x = x.rotate(float(rot))
			x = x.resize((64,64))
			#print(x.size)

			shape = 3, x.size[0], x.size[1]
			x = np.array(x, np.float32, copy=False)
			#print(x.shape)
			x = 1.0 - torch.from_numpy(x)
			#print(x.shape)
			x = x.transpose(0, 1).contiguous().view(shape)
			#print(x.size)
			return x
		except:
		    pass

	


	

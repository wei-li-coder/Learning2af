import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import cv2

# rootpath for dataset
root ='/data/wl/autofocus/learn2focus/dataset/'

def default_loader(path):
	return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# make my dataset
class MyDataset(Dataset):
	# initialize
	def __init__(self,txt, transform=None,target_transform=None, loader=default_loader):
		super(MyDataset,self).__init__()
		fh = open(txt, 'r')
		imgs = []
		for line in fh: 
			line = line.strip('\n')
			line = line.rstrip('\n')
			words = line.split()
			# patch path and label
			imgs.append((words[0],int(words[1])))      
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader        
        
	def __getitem__(self, index):
		fn, label = self.imgs[index]
		img = self.loader(fn)
		# focal stack
		# to stack all channel together
		# img_new = img[:,:128]
		# for i in range(1,98):
		# 	img_new = np.dstack((img_new, img[:,128*i:128*(i+1)]))
		img = np.dstack((img[:,:128],img[:,128:]))
		# single slice
		# all-zero tensor except observed channel
		channel_idx = int(fn[-6:-4])
		img_zeros = np.zeros((128,128), dtype=np.float32)
		if channel_idx == 0:
			for _ in range(96):
				img = np.dstack((img, np.zeros((128,128), dtype=np.float32)))
		elif channel_idx == 1:
			img = np.dstack((img_zeros, img_zeros, img))
			for _ in range(94):
				img = np.dstack((img, np.zeros((128,128), dtype=np.float32)))
		else:
			for _ in range(2*channel_idx-1):
				img_zeros = np.dstack((img_zeros, np.zeros((128,128), dtype=np.float32)))
			img = np.dstack((img_zeros, img))
			for _ in range(96 - 2*channel_idx):
				img = np.dstack((img, np.zeros((128,128), dtype=np.float32)))
		# img = img_new
		
		if self.transform is not None:
			img = self.transform(img)
			
		return img, label#return
	
	def __len__(self):
		return len(self.imgs)
 

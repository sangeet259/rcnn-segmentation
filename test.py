import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils import data

from loaders.pascal_voc_loader import pascalVOCLoader
from models.segmentor import rcnn
import os
import logging
from sklearn.metrics import jaccard_similarity_score as jsc

def initialize_rcnn(n_classes):

	segmentor = rcnn

	try:
		segmentor = segmentor()
	except Exception as e:
		print('Error occured in initialising rcnn ' + str(e))
		sys.exit(1)

	return segmentor



batch_size = 32
use_gpu = torch.cuda.is_available()

segmentor = initialize_rcnn(21)
segmentor.load_state_dict(torch.load('./checkpoints/0-19-0.5117.pyt'))

if use_gpu:
	segmentor.cuda()


def test():
	jsc_list=[]

	data_loader = pascalVOCLoader
	data_path = "/home/temp_siplab/Datasets/VOCdevkit/VOC2012/"

	loader = data_loader(data_path, is_transform=True,split="train",
		img_size=(256, 256),ohe=False)

	n_classes = loader.n_classes
	testloader = data.DataLoader(loader, batch_size=batch_size, num_workers=4, shuffle=True)

	os.system("mkdir -p checkpoints")

	for i, (images, labels) in enumerate(testloader):

		if use_gpu:
			images = Variable(images.cuda())
			labels = Variable(labels.cuda())
		else:
			images = Variable(images)
			labels = Variable(labels)

		segmentor.zero_grad()

		output = segmentor(images)
		lbl = labels.cpu().numpy().reshape(-1)
		target = torch.max(output,dim=1)[1].cpu().detach().numpy().reshape(-1)

		jsc_list.append(jsc(target,lbl))
	print(np.mean(jsc_list))


		

if __name__ == '__main__':
	test()

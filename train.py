import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

from torch.autograd import Variable
from torch.utils import data

from loaders.pascal_voc_loader import pascalVOCLoader
from loss import cross_entropy2d
from models.segmentor import rcnn
from models.discriminator import LargeFOV
import pudb
import os
import time
# from logger import Logger
import logging
logging.basicConfig(filename='loss.log',level=logging.INFO)

def initialize_rcnn(n_classes):

	segmentor = rcnn

	try:
		segmentor = segmentor()
	except Exception as e:
		print('Error occured in initialising rcnn ' + str(e))
		sys.exit(1)

	return segmentor



batch_size = 96
elasped_time = 0
use_gpu = torch.cuda.is_available()

# pu.db

segmentor = initialize_rcnn(21)
segmentor.load_state_dict(torch.load('./checkpoints/0-19-0.5117.pyt'))

if use_gpu:
	segmentor.cuda()

# seg_optim = optim.RMSprop(segmentor.parameters(), lr=1e-7, momentum=0.9)
seg_optim = optim.Adam(segmentor.parameters(), lr=1e-7)


def train(epochs):
	best_loss=100000
	best_ep_loss = 0.72


	data_loader = pascalVOCLoader
	data_path = "/home/temp_siplab/Datasets/VOCdevkit/VOC2012/"

	loader = data_loader(data_path, is_transform=True,img_size=(256, 256),ohe=False)

	n_classes = loader.n_classes
	trainloader = data.DataLoader(loader, batch_size=batch_size, num_workers=4, shuffle=True)

	os.system("mkdir -p checkpoints")

	for epoch in range(epochs):
		t0 = time.time()
		ep_loss = 0
		count = 0

		for i, (images, labels) in enumerate(trainloader):
			try:

				if use_gpu:
					images = Variable(images.cuda())
					labels = Variable(labels.cuda())
				else:
					images = Variable(images)
					labels = Variable(labels)
				# import pudb;pu.db
				segmentor.zero_grad()

				output = segmentor(images)

				# out,tar, value = torch.max(output,dim=1)
				pu.db

				loss = nn.functional.cross_entropy(output, labels)
				ep_loss += loss.item()
				loss.backward()

				seg_optim.step()
				
				count = i


				if i % 5 == 4:    # print every 5 mini-batches

					print('[%d, %5d] loss: %.4f' %
						  (epoch + 1, i + 1, loss))

					if loss.detach().cpu().numpy() < best_loss:
						np.save("best_loss.npy",loss.detach())
						ckpt_path = "checkpoints/{0}-{1}-{2:0.4f}.pyt".\
						format(epoch,i,(loss.detach().cpu().numpy()))

						print("Better loss (= {0:0.4f}) found,\
							saving model at {1}".format(loss,ckpt_path))

						best_loss = loss.detach().cpu().numpy()
						torch.save(segmentor.state_dict(),ckpt_path)

			except KeyboardInterrupt:
				import IPython; IPython.embed()

			except Exception as e:
				print(str(e) + "Droppping to IPython Shell,\
					so save your model while it's time ^_^ ")

			# finally:
			# 	sys.exit()

		t1 = time.time()
		print("Time/Epoch = "+ str(t1-t0))
		ep_loss = ep_loss/count
		print("Epoch loss : {}".format(ep_loss))
		if ep_loss < best_ep_loss:
			ep_ckpt_path = "checkpoints/epochs/{0}-{1:0.4f}.pyt".\
			format(epoch,(ep_loss))
			torch.save(segmentor.state_dict(),ep_ckpt_path)
			best_ep_loss = ep_loss


		

if __name__ == '__main__':
	epochs = 200
	train(epochs)

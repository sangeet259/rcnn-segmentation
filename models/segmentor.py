#Originally written by https://github.com/meetshah1995/pytorch-semseg

import torch.nn as nn
import torch.nn.functional as F

class rcnn(nn.Module):

	def __init__(self,K=128):
		super(rcnn, self).__init__()

		self.bn1 = nn.BatchNorm2d(3)
		self.max_pool = nn.MaxPool2d(3,2)
		# self.lrn = nn.LocalResponseNorm(13)
		# self.droput = nn.Dropout(droput_p)
		self.relu = nn.ReLU()

		self.conv1 = nn.Conv2d(3, K, 5,1)

		self.rcl_1_feed_fwd = nn.Conv2d(K,K,3,1,1)
		self.rcl_1_rec = nn.Conv2d(K,K,3,1,1)

		self.bn2 = nn.BatchNorm2d(K)
		self.conv2 = nn.Conv2d(K,K,3,1,1)
		
		self.rcl_2_feed_fwd = nn.Conv2d(K,K,3,1,1)
		self.rcl_2_rec = nn.Conv2d(K,K,3,1,1)

		self.bn3 = nn.BatchNorm2d(K)
		self.conv2 = nn.Conv2d(K,K,3,1,1)
		
		self.rcl_3_feed_fwd = nn.Conv2d(K,K,3,1,1)
		self.rcl_3_rec = nn.Conv2d(K,K,3,1,1)

		self.bn4 = nn.BatchNorm2d(K)
		self.conv3 = nn.Conv2d(K,K,3,1,1)
		
		self.rcl_4_feed_fwd = nn.Conv2d(K,K,3,1,1)
		self.rcl_4_rec = nn.Conv2d(K,K,3,1,1)

		self.deconv1 = nn.ConvTranspose2d(128,128,3,2)
		self.deconv2 = nn.ConvTranspose2d(128,21,3,2)
		self.deconv3 = nn.ConvTranspose2d(21,21,5,1)

		self.linear = nn.Linear(K,10)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):

		out = self.conv1(self.bn1(x))
		# pu.db
		out = self.max_pool(out)

		# First RCL
		out_f = out_r = self.rcl_1_feed_fwd(out)
		for i in range(3):
			out_r = self.rcl_1_rec(out_r) + out_f
		out = out_r
		out = self.bn2(self.relu(out))
		# out = self.droput(out)

		# Second RCL
		out_f = out_r = self.rcl_2_feed_fwd(out)
		for i in range(3):
			out_r = self.rcl_2_rec(out_r) + out_f

		out = out_r
		out = self.bn3(self.relu(out))
		out_second_rcl = out
		out = self.max_pool(out)
		# out = self.droput(out)

		# Third RCL 
		out_f = out_r = self.rcl_3_feed_fwd(out)
		for i in range(3):
			out_r = self.rcl_3_rec(out_r) + out_f
		out = out_r
		out = self.bn4(self.relu(out))
		# out = self.droput(out)

		# Fourth RCL
		out_f = out_r = self.rcl_4_feed_fwd(out)
		for i in range(3):
			out_r = self.rcl_4_rec(out_r) + out_f

		out = out_r
		out = self.relu(out)
		# out = self.droput(out)
		out = self.deconv1(out)
		out += out_second_rcl
		out = self.deconv2(out)
		out = self.deconv3(out)
		out = nn.functional.upsample(out,size=256,mode='bilinear')
		return out
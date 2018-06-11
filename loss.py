import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pudb


def cross_entropy2d(input, target, weight=None, size_average=True):
	pu.db
	n, c, h, w = input.size()
	nt, ht, wt = target.size()

	# Handle inconsistent size between input and target
	if h > ht and w > wt: # upsample labels
		target = target.unsequeeze(1)
		target = F.upsample(target, size=(h, w), mode='nearest')
		target = target.sequeeze(1)
	elif h < ht and w < wt: # upsample images
		input = F.upsample(input, size=(ht, wt), mode='bilinear')
	elif h != ht and w != wt:
		raise Exception("Only support upsampling")

	log_p = F.log_softmax(input, dim=1)
	log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
	log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
	log_p = log_p.view(-1, c)

	mask = target >= 0
	target = target[mask]
	loss = F.nll_loss(log_p, target, ignore_index=250,
					  weight=weight, size_average=False)
	if size_average:
		loss /= mask.float().data.sum()
	return loss

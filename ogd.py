import numpy as np
import torch
from torch.autograd import Variable

'''
Implementation of Online Gradient Descent step.

Input: 	x_t - current point in the decision set K
		f_t - loss function incurred
		step - step size
		project - a projection oracle onto set K

Returns:	tuple of next point as a torch tensor, and the loss incurred.
'''
def ogd_step(x_t, f_t, step, project):
	x_t = Variable(x_t, requires_grad=True)
	loss = f_t(x_t)
	loss.backward()
	gradient = x_t.grad.data
	new_point = x_t.data - step * gradient
	updated = project(new_point.numpy())
	return torch.tensor(updated), float(loss.data.numpy())

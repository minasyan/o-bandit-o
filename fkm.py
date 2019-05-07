import numpy as np
import torch
from torch.autograd import Variable

'''
Sample a random unit vector in d dimensions by normalizing i.i.d gaussian vector

Input:	d - dimensions
'''
def sample_random_unit(d):
	gaussian = np.random.normal(size=d)
	unit = gaussian / np.linalg.norm(gaussian)
	return torch.tensor(unit)

'''
Implementation of the FKM algorithm (OGD without a gradient).

Input:	x_t - current point in the decision set K_delta
		f_t - loss function incurred
		d - dimensionality of the feasible set
		delta - perturbation parameter
		step - step size
		project - a projection oracle onto set K_delta

Returns:	tuple of next point as a torch tensor, and the loss incurred.
'''
def fkm_step(x_t, f_t, d, delta, step, project):
	u_t = sample_random_unit(d)
	y_t = x_t + delta * u_t
	loss = float(f_t(y_t).data.numpy())
	g_t = (d * loss / delta) * u_t
	new_point = x_t - step * g_t
	update = project(new_point.numpy())
	return torch.tensor(update), loss

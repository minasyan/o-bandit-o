import numpy as np
import torch
from ogd import ogd_step
from fkm import fkm_step

'''
Linear function that returns the dot product with a random input tensor.
'''
def f_linear(G, d):
	in_tensor = torch.tensor(np.random.normal(size=d), dtype=torch.float64)
	in_tensor = projection(G)(in_tensor)
	def f_t(x):
		return torch.dot(in_tensor, x)
	return f_t

'''
Quadratic function that returns the matrix form quadratic with a random matrix.
'''
#def f_quadratic(G, d):


'''
Projection oracle onto a ball of radius r.
'''
def projection(r):
	def project(x):
		if np.linalg.norm(x) <= r:
			return x
		return (r / np.linalg.norm(x)) * x
	return project

'''
Run the online algorithm to optimize a function offline, i.e.
feed the same input at all iterations.
The decision set is a ball of radius R, so the diameter is D=2R.
G is the bound on the gradients. T is the number of iterations.
f is the function template we are running optimization on.
'''
def run_offline(T, G, R, d, f):
	#import pdb; pdb.set_trace()

	ogd_x = torch.zeros(d, dtype=torch.float64)
	fkm_x = torch.zeros(d, dtype=torch.float64)
	f_opt = f(G, d)
	D = 2 * R
	C = G * R
	ogd_stepsize = D / (G * T**(1/2))
	fkm_stepsize = D / (C * d * T**(3/4))
	delta = 1 / T**(1/4)
	ogd_project = projection(R)
	fkm_project = projection((1-delta)*R)
	ogd_loss = 0
	fkm_loss = 0
	for t in range(T):
		print("Iteration {0}".format(t))
		ogd_x, loss = ogd_step(ogd_x, f_opt, ogd_stepsize, ogd_project)
		ogd_loss += loss
		print("OGD action is {0}, the average loss so far is {1}".format(ogd_x.data, ogd_loss/(t+1)))
		fkm_x, loss = fkm_step(fkm_x, f_opt, d, delta, fkm_stepsize, fkm_project)
		fkm_loss += loss
		print("FKM action is {0}, the average loss so far is {1}".format(fkm_x.data, fkm_loss/(t+1)))

'''
Run the online algorithm to loss over random samples, i.e.
feed a function with a random input at each iteration.
The decision set is a ball of radius R, so the diameter is D=2R.
G is the bound on the gradients. T is the number of iterations.
f is the function template we are running optimization on.
'''
def run_random(T, G, R, d, f):
	#import pdb; pdb.set_trace()

	ogd_x = torch.zeros(d, dtype=torch.float64)
	fkm_x = torch.zeros(d, dtype=torch.float64)
	D = 2 * R
	C = G * R
	ogd_stepsize = D / (G * T**(1/2))
	fkm_stepsize = D / (C * d * T**(3/4))
	delta = 1 / T**(1/4)
	ogd_project = projection(R)
	fkm_project = projection((1-delta)*R)
	ogd_loss = 0
	fkm_loss = 0
	for t in range(T):
		print("Iteration {0}".format(t))
		current_f = f(G, d)
		ogd_x, loss = ogd_step(ogd_x, current_f, ogd_stepsize, ogd_project)
		ogd_loss += loss
		print("OGD action is {0}, the average loss so far is {1}".format(ogd_x.data, ogd_loss/(t+1)))
		fkm_x, loss = fkm_step(fkm_x, current_f, d, delta, fkm_stepsize, fkm_project)
		fkm_loss += loss
		print("FKM action is {0}, the average loss so far is {1}".format(fkm_x.data, fkm_loss/(t+1)))

if __name__ == '__main__':
	d = 10
	T = 1000
	G = 5.0
	R = 3.0
	run_random(T, G, R, d, f_linear)

import numpy as np
import torch
from ogd import ogd_step
from fkm import fkm_step
import matplotlib.pyplot as plt

'''
Linear function that returns the dot product with a random input tensor.
'''
def f_linear(G, d, R):
	in_tensor = torch.tensor(np.random.normal(1, 1, size=d), dtype=torch.float64)
	in_tensor = projection(G)(in_tensor)
	def f_t(x):
		return torch.dot(in_tensor, x)
	return f_t

'''
Quadratic function that returns the matrix quadratic form with a random matrix
plus a dot product with a random unit vector.
'''
def f_quadratic(G, d, R):
	in_matrix = np.random.normal(1, 1, size=(d, d))
	norm_bound = np.linalg.norm(in_matrix.T+in_matrix, ord=2)
	if norm_bound > (G-1)/R:
		in_matrix = in_matrix * (G-1)/(R*norm_bound)
	in_matrix = torch.tensor(in_matrix, dtype=torch.float64)
	in_tensor = torch.tensor(np.random.normal(1, 1, size=d), dtype=torch.float64)
	in_tensor = projection(1)(in_tensor)
	def f_t(x):
		return torch.mm(x.view(1, -1), torch.mm(in_matrix, x.view(-1, 1)))[0][0] + torch.dot(in_tensor, x)
	return f_t


'''
Projection oracle onto a ball of radius r.
'''
def projection(r):
	def project(x):
		if np.linalg.norm(x) <= r:
			return x
		return (r / np.linalg.norm(x)) * x
	return project

def plot_runs(opt_name, f_name, ogd_losses, fkm_losses, T):
	plt.rcParams.update({'font.size': 12})
	xs = [t for t in range(1, T+1)]
	title = opt_name+" for "+f_name
	plt.plot(xs, ogd_losses, 'r-', label='OGD')
	plt.plot(xs, fkm_losses, 'b-', label='FKM')
	plt.legend(loc='upper right', frameon=False)
	plt.xlabel("Iterations")
	plt.ylabel("Average loss incurred")
	#plt.ylim(top=0.6)
	plt.title(title)
	plt.show()


'''
Run the online algorithm to optimize a function offline, i.e.
feed the same input at all iterations.
The decision set is a ball of radius R, so the diameter is D=2R.
G is the bound on the gradients. T is the number of iterations.
f is the function template we are running optimization on.
'''
def run_offline(T, G, R, d, f, name):
	#import pdb; pdb.set_trace()

	ogd_x = torch.zeros(d, dtype=torch.float64)
	fkm_x = torch.zeros(d, dtype=torch.float64)
	f_opt = f(G, d, R)
	D = 2 * R
	C = G * R
	ogd_stepsize = D / (G * T**(1/2))
	fkm_stepsize = D / (C * d * T**(3/4))
	delta = 1 / T**(1/4)
	ogd_project = projection(R)
	fkm_project = projection((1-delta)*R)
	ogd_loss = 0
	fkm_loss = 0
	ogd_all_losses = []
	fkm_all_losses = []
	for t in range(T):
		#print("Iteration {0}".format(t))
		ogd_x, loss = ogd_step(ogd_x, f_opt, ogd_stepsize, ogd_project)
		ogd_loss += loss
		#print("OGD action is {0}, the average loss so far is {1}".format(ogd_x.data, ogd_loss/(t+1)))
		ogd_all_losses.append(ogd_loss/(t+1))
		fkm_x, loss = fkm_step(fkm_x, f_opt, d, delta, fkm_stepsize, fkm_project)
		fkm_loss += loss
		#print("FKM action is {0}, the average loss so far is {1}".format(fkm_x.data, fkm_loss/(t+1)))
		fkm_all_losses.append(fkm_loss/(t+1))
	plot_runs("Offline optimization", name, ogd_all_losses, fkm_all_losses, T)


'''
Run the online algorithm to loss over random samples, i.e.
feed a function with a random input at each iteration.
The decision set is a ball of radius R, so the diameter is D=2R.
G is the bound on the gradients. T is the number of iterations.
f is the function template we are running optimization on.
'''
def run_random(T, G, R, d, f, name):
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
	ogd_all_losses = []
	fkm_all_losses = []
	for t in range(T):
		#print("Iteration {0}".format(t))
		current_f = f(G, d, R)
		ogd_x, loss = ogd_step(ogd_x, current_f, ogd_stepsize, ogd_project)
		ogd_loss += loss
		#print("OGD action is {0}, the average loss so far is {1}".format(ogd_x.data, ogd_loss/(t+1)))
		ogd_all_losses.append(ogd_loss/(t+1))
		fkm_x, loss = fkm_step(fkm_x, current_f, d, delta, fkm_stepsize, fkm_project)
		fkm_loss += loss
		#print("FKM action is {0}, the average loss so far is {1}".format(fkm_x.data, fkm_loss/(t+1)))
		fkm_all_losses.append(fkm_loss/(t+1))
	plot_runs("Online Random Optimization", name, ogd_all_losses, fkm_all_losses, T)

if __name__ == '__main__':
	d = 10
	T = 5000
	G = 5.0
	R = 3.0
	run_random(T, G, R, d, f_quadratic, "Quadratic function")

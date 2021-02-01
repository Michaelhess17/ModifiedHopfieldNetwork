import numpy as np
import random
import hdnet.hopfield as hdn
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
import scipy.io as spio
import math
import scipy.special as sps
import scipy.optimize as spo
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from collections import deque
from sklearn.metrics import confusion_matrix


# TODO make data_type auto select, so that a folder can have multiple data types
# TODO print units for StimTimes and Ts when loading data

class ModifiedHopfieldNet:
	"""
		Argumentss:
		N = number of nodes to build Hopfield network with
		in_directory = the relative path to a folder containing the raw .mat files
		out_directory = the relative path to a folder that you want to store the python data. If not used, defaults to in_directory
		splits: the number of times to split the data to train on each portion (slices sequentially
			--> default: 3 splits = beginning, middle, end thirds of experiment)
		train_percent = the percentage of each chunk of data (the number of chunks as defined by splits) that will be
			used to train a Hopfield network --> default: 0.66
		num_nets: The number of replications to use when training Hopfield networks --> default: 5
		exp_type = 'map' to conduct a MAP estimate analysis, J to analyze the changing connectivity matrices through time.
			--> default: 'J'
	"""

	def __init__(self, in_directory, out_directory=None, splits=3, train_percent=0.66, num_nets=50, exp_type='J',
	             data_type='sim', dt=1600, n_jobs=1, N=None, num_bs=1, threshold=250, stim_shift=0):
		self.in_directory = in_directory
		self.requested_files = None
		self.type = exp_type
		if out_directory is None:
			self.out_directory = self.in_directory
		else:
			self.out_directory = out_directory
		self.dt = dt
		self.dats = []
		self.filenames = []
		if N is None:
			self.N = 0
			self.N = self.get_N()
		else:
			self.N = N
		self.splits = splits
		self.experiments = []
		self.train_percent = train_percent
		self.num_nets = num_nets
		self.networks = []
		self.data_type = data_type
		self.n_jobs = n_jobs
		self.num_bs = num_bs
		self.threshold = threshold
		self.bs_inds = []
		self.filenames = []
		self.bs_inds = []
		self.thetas = []
		self.Js = []
		self.stim_shift = stim_shift
		if (self.n_jobs == 1) & (self.type != 'MI') & (self.type != 'map'):
			self.load_and_save_data(dt=self.dt)
		if (self.n_jobs > 1) & (self.type == 'J'):
			files = []
			for file in os.listdir(self.in_directory):
				filename = file[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
				if filename not in os.listdir(self.out_directory):
					files.append(file)
			p = Pool(self.n_jobs)
			p.map(self.run_multiprocessing_J, files)
			p.close()
			p.join()

	def get_N(self):
		dats, _ = self.get_dats(get_N=True)
		try:
			Cs = dats[0][0]['Cs']
		except:
			Cs = dats[0][0]['data']['Cs']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
		N = np.max(Cs) + 1
		if self.type in ['map', 'MAP', 'MI', 'mi']:
			N += 1
		return N

	def load_and_save_data(self, **kwargs):
		input_files = np.asarray(os.listdir(self.in_directory))
		print("The following files are in your input directory: \n")
		for k, file in enumerate(input_files):
			print(f'File {k}: {file}')
		requested_files = input('Please enter the index of files your would like to analyze separated by a comma. \n '
		                        'For example, if you wanted the first and third file, type "0, 2" \n '
		                        'If you would like to use all listed files, just press Enter')
		if requested_files == '':
			requested_files = input_files
		else:
			requested_files = requested_files.split(',')
			for k, file in enumerate(requested_files):
				requested_files[k] = int(file)
			requested_files = input_files[requested_files]
		for file in requested_files:
			filename = file[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
			if filename not in os.listdir(self.out_directory):
				print(f'---------------- The file {filename} was not found in the out directory  -----------------')
				print(f'---------------------- Importing .mat file: {file} instead ----------------------')
				dat = spio.loadmat(os.path.join(self.in_directory, file))
				ys = self.binaryVecs(dat, **kwargs)
				self.experiments.append(ys)
				y_sparse = csr_matrix(ys, dtype='uint8')
				save_npz(os.path.join(self.out_directory, filename), y_sparse)
			else:
				print(f'------------------ Loading file: {filename} from out directory --------------------')
				try:
					ys = load_npz(os.path.join(self.out_directory, filename)).toarray()
				except:
					dat = spio.loadmat(os.path.join(self.in_directory, file))
					ys = self.binaryVecs(dat, **kwargs)
					y_sparse = csr_matrix(ys, dtype='uint8')
					save_npz(os.path.join(self.out_directory, filename), y_sparse)
				self.experiments.append(ys)

	def build_and_train_networks(self):
		all_accs = []
		for i, memories in enumerate(self.experiments):
			accs = []
			print(f'---------------------- Conducting experiment: {i} ----------------------')
			memories_chunked = self.chunked(memories, self.splits)
			experiment_nets = []
			for j, memory_chunk in enumerate(memories_chunked):
				avg_acc = []
				chunked_nets = []
				for _ in tqdm(range(self.num_nets)):
					hop = hdn.HopfieldNetMPF(N=self.N)
					rand_memories = np.array(
						[random.choice(memory_chunk) for _ in range(round(len(memory_chunk) * self.train_percent))])
					hop.store_patterns_using_mpf(rand_memories + 0.0)
					if self.type == 'map':
						avg_acc.append(self.get_accuracy(memory_chunk, hop, precision_recall=False))
					chunked_nets.append(hop)
				experiment_nets.append(chunked_nets)
				accs.append(avg_acc)
				if self.type == 'map':
					print(f'Experiment: {i} // Chunk: {j} // Avg Accuracy: {round(np.mean(avg_acc), 3)} +/- '
					      f'{round(np.std(avg_acc), 3)}')
				else:
					print(f'Experiment: {i} // Chunk: {j}')
			all_accs.append(accs)
			print(f'---------------------- Finished experiment: {i} ----------------------')
			self.networks.append(experiment_nets)
		return all_accs

	@staticmethod
	def explicitMLE(means, corrs):
		n = len(means)
		if n > 20:
			raise ValueError('Cannot peform fitting when N>20. Reduce N and try again!')

		#
		def logL(x):
			J = np.reshape(x[:n ** 2], [n, n])
			h = x[n ** 2:]
			# get first term
			Z = 0
			for i in range(np.power(2, n)):
				x = np.asarray(list([int(j) for j in np.binary_repr(i, width=n)]))
				E = -np.inner(h, x) + np.inner(x, np.dot(J, x))
				Z += np.exp(E)
			# combine into logL
			logL = -np.sum(means * h) + np.sum(corrs * J) - np.log(Z)
			return logL

		# 1. Do MLE fit
		# For now, assume contrastive divergence unnecessary
		# record log likelihood and actual J's and h's
		def jac_MLE(x):
			J = np.reshape(x[:n ** 2], [n, n])
			h = x[n ** 2:]
			#
			moments_model = np.zeros(n ** 2 + n)
			Z = 0
			for i in range(np.power(2, n)):
				x = np.asarray(list([int(j) for j in np.binary_repr(i, width=n)]))
				E = -np.inner(h, x) + np.inner(x, np.dot(J, x))
				Z += np.exp(E)
				moments_model[:n ** 2] += np.exp(E) * np.reshape(np.outer(x, x), [n ** 2])
				moments_model[n ** 2:] += -np.exp(E) * x
			moments_model /= Z
			moments_data = np.hstack([np.reshape(corrs, n ** 2), -means])
			return moments_data - moments_model

		foo_MLE = spo.minimize(lambda x: -logL(x), x0=np.random.uniform(size=n ** 2 + n), jac=lambda x: -jac_MLE(x))
		logL_MLE = -foo_MLE.fun;
		J_guess = foo_MLE.x[:n ** 2];
		h_guess = foo_MLE.x[n ** 2:]
		return h_guess, np.reshape(J_guess, [n, n])

	def contrastiveDivergence(self, means, corrs, alpha=0.1, thresh=0.05):
		n = len(means)
		# choose initial J and theta
		J_guess = np.random.uniform(size=[n, n])
		theta_guess = np.random.uniform(size=n)

		grad_norm = thresh * 2
		while grad_norm > thresh:
			# do gradient ascent
			# get better estimate of gradient
			mean_model = np.zeros(n)
			corr_model = np.zeros([n, n])
			for k in range(10):
				foo = self.monte_carlo(J_guess, theta_guess, n, k=0.1)
				mean_model += foo[0]
				corr_model += foo[1]
			mean_model /= 10
			corr_model /= 10
			# gradient ascent
			grad_norm = np.sum(np.abs(means - mean_model) / n) + np.sum(np.abs(corrs - corr_model) / n ** 2)
			theta_guess += alpha * (means - mean_model)
			J_guess += alpha * (corrs - corr_model)
		return theta_guess, J_guess

	@staticmethod
	def monte_carlo(J, theta, n, k=0.1):
		# implement MonteCarlo to evaluate gradient and do gradient ascent
		# do this how many times?

		xold = np.asarray([int(i) for i in np.binary_repr(np.random.randint(0, np.power(2, n)), n)])
		Eold = np.inner(theta, xold) + np.inner(xold, np.dot(J, xold))
		# collect new samples and calculate means and correlations during.
		burn_in = 5000
		tot_time = int(30000)
		pm = np.exp(-k * np.arange(1, n))
		pm /= np.sum(pm)
		mean = np.zeros(n)
		corr = np.zeros([n, n])
		for t in range(burn_in + tot_time):
			# generate candidate
			# we'll look at all configurations that flip one bit, could modify this.
			m = np.random.choice(n - 1, size=1, p=pm) + 1
			# m gives the number of bits we flip
			foo = np.random.choice(n, size=m, replace=False)
			xnew = np.zeros(n)
			for i in range(n):
				if i in foo:
					xnew[i] = 1 - xold[i]
				else:
					xnew[i] = xold[i]
			# calculate the acceptance ratio
			Enew = np.inner(theta, xnew) + np.inner(xnew, np.dot(J, xnew))
			dE = Enew - Eold
			acceptance_ratio = np.exp(-dE)
			# accept or reject
			u = np.random.uniform()
			if u < acceptance_ratio:
				Eold = Enew
				if t > burn_in - 1:
					mean += xnew
					corr += np.outer(xnew, xnew)
				xold = xnew
			else:
				if t > burn_in - 1:
					mean += xold
					corr += np.outer(xold, xold)
		mean /= tot_time
		corr /= tot_time
		return mean, corr

	def get_means_and_corrs(self, binaryVec):
		means = np.mean(binaryVec, axis=0)
		corrs = np.dot(binaryVec, binaryVec.T)
		return means, corrs

	def run_explicitMLE(self):
		for i, memories in enumerate(self.experiments):
			print(f'---------------------- Conducting experiment: {i} ----------------------')
			memories_chunked = self.chunked(memories, self.splits)
			for j, memory_chunk in enumerate(memories_chunked):
				for _ in tqdm(range(self.num_nets)):
					rand_memories = np.array(
						[random.choice(memory_chunk) for _ in range(round(len(memory_chunk) * self.train_percent))])
					means, corrs = self.get_means_and_corrs(rand_memories)
					theta, J = self.explicitMLE(means, corrs)
					self.thetas.append(theta)
					self.Js.append(J)
			print(f'---------------------- Finished experiment: {i} ----------------------')
		return self.thetas, self.Js

	def run_bootstrap_Js(self, bootstrap_sizes):
		for size in bootstrap_sizes:
			bs_nets = []
			for bootstrap_num in range(self.num_bs):
				inds = np.random.choice(range(self.N), size, replace=False)
				self.bs_inds.append(inds)
				for i, memories in enumerate(self.experiments):
					memories = memories[:, inds]
					print(f'---------------------- Conducting experiment: {i} ----------------------')
					memories_chunked = self.chunked(memories, self.splits)
					experiment_nets = []
					for j, memory_chunk in enumerate(memories_chunked):
						chunked_nets = []
						for _ in range(self.num_nets):
							hop = hdn.HopfieldNetMPF(N=size)
							rand_memories = np.array(
								[random.choice(memory_chunk) for _ in
								 range(round(len(memory_chunk) * self.train_percent))])
							hop.store_patterns_using_mpf(rand_memories + 0.0)
							chunked_nets.append(hop)
						experiment_nets.append(chunked_nets)
						print(f'Experiment: {i}/{len(self.experiments)} // Chunk: {j}/{self.splits} // Bs Size: {size}')
					print(f'---------------------- Finished experiment: {i} ----------------------')
					bs_nets.append(experiment_nets)
			self.networks.append(bs_nets)

	def chunked(self, iterable, n):
		chunksize = int(math.ceil(len(iterable) / n))
		return (iterable[i * chunksize:i * chunksize + chunksize] for i in range(n))

	def getL(self, x, h1, A):
		L = np.exp(np.dot(h1.T, x) + A)
		if L > 1:
			return 1
		else:
			return 0

	def get_preds(self, y, hop):
		J = hop.J
		h = -hop.theta
		A = J[-1, -1]
		B = h[-1]
		# J0 = J[:-1, :-1]
		j = J[-1, :-1]
		# jT = J[-1, :-1]
		# J = J0
		h1 = 2 * j
		# h0 = h
		A = A + B
		x = y[:-1]
		p = self.getL(x, h1, A)
		return p

	def get_accuracy(self, memories, hop, precision_recall=False):
		accuracy = 0
		y_preds = []
		y_true = []
		for k, i in enumerate(memories):
			y_preds.append(self.get_preds(i, hop))
			y_true.append(i[-1])
			if y_preds[k] == y_true[k]:
				accuracy += 1
		accuracy = accuracy / len(memories)
		if not precision_recall:
			return round(accuracy * 100, 3)
		else:
			tn, fp, fn, tp = confusion_matrix(y_true, y_preds)
			return tn, fp, fn, tp, accuracy

	def get_js(self, filename='Js_Joost.pkl'):
		Js = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for networks in memory_chunk_networks:
					if type(networks) != list:
						chunk_nets.append(networks._J)
					else:
						bs_nets = []
						for network in networks:
							bs_nets.append(network._J)
						chunk_nets.append(bs_nets)
				experiment_nets.append(chunk_nets)
			Js.append(experiment_nets)
		Js = np.array(Js).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(Js, file)
		return Js

	def get_thetas(self, filename='Thetas_Joost_old.pkl'):
		thetas = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for networks in memory_chunk_networks:
					if type(networks) != list:
						chunk_nets.append(networks._theta)
					else:
						bs_nets = []
						for network in networks:
							bs_nets.append(network._theta)
						chunk_nets.append(bs_nets)
				experiment_nets.append(chunk_nets)
			thetas.append(experiment_nets)
		thetas = np.array(thetas).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(thetas, file)
		return thetas

	def binaryVecs(self, dat, dt=None):
		if ((self.type == 'map') or (self.type == 'MI')) & (self.data_type == 'old'):
			print("Since you're using the old data type, the following adjustments will be made: \n")
			print("The StimTimes variable will be assumed to be in the sample number form \n"
			      "The Ts variable will also be assumed to be in the sample number form \n"
			      "No changes will be made to either of the raw variables")
			if dt is None:
				dt = 0.05
			StimTimes = dat['StimTimes']
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in StimTimes])
			Tmax = np.max([np.max(StimTimes), np.max(Ts)])  # using just Ts will be more accurate
		elif (self.type == 'J') & (self.data_type == 'old'):
			if dt is None:
				dt = 800
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			# Cs, Ts = self.clean_Cs_and_Ts(Cs, Ts)
			Tmax = np.max(Ts)
		elif (self.type == 'J') & (self.data_type == 'sim'):
			ys = dat['SPK_Times'].T
			return ys
		elif ((self.type == 'map') or (self.type == 'MI')) & (self.data_type == 'new'):
			print("Since you're using the new data type, the following adjustments will be made: \n")
			print("The StimTimes variable will be assumed to be in seconds \n"
			      "To get the sample number, we multiply StimTimes by 16,000 Hz \n"
			      "The Ts variable will be assumed to already be in sample number form, so no changes will be made to it")
			if dt is None:
				dt = 800
			dat = dat['data']
			StimTimes = dat['StimTimes']
			Cs = dat['Cs']
			Ts = dat['Ts']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
			Ts = np.array([a[0] for a in Ts.tolist()[0][0].tolist()], dtype='uint32')
			StimTimes = np.array([a[0]*16000 for a in StimTimes.tolist()[0][0].tolist()], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in StimTimes])
			Tmax = np.max([np.max(StimTimes), np.max(Ts)])  # using just Ts will be more accurate
		else:
			if dt is None:
				dt = 800
			Cs = dat['data']['Cs']
			Ts = dat['data']['Ts']
			Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
			Ts = np.array([a[0] for a in Ts.tolist()[0][0].tolist()], dtype='uint32')
			CsTs = self.clean_Cs_and_Ts(Cs, Ts)
			Cs = CsTs[0]
			Ts = CsTs[1]
			Tmax = np.max(Ts)

		foo_x = np.asarray([int(i / dt) for i in Ts])

		ys = []
		# if self.stim_shift < 0:
		# 	stim_hist = deque(maxlen=np.abs(self.stim_shift))
		# 	for _ in range(self.stim_shift):
		# 		stim_hist.append(0)

		for i in range(int(Tmax / dt)-self.stim_shift):
			if i in foo_x:
				# which neurons are firing
				inds = (i * dt <= Ts) * (Ts < (i + 1) * dt)
				neurons = Cs[inds]
				# for neuron in neurons:
				# 	counter[neuron] += 1
				neurons = list(set(neurons))
				foo2 = np.zeros(self.N)
				foo2[neurons] = 1
			# foo2[foo2 == 0] = np.nan
			else:
				foo2 = np.zeros(self.N)
			# foo2[foo2 == 0] = np.nan
			# is the stimulus firing
			if (self.type == 'map') or (self.type == 'MI'):
				# if self.stim_shift < 0:
				# 	foo2[-1] = stim_hist[0]
				# if i in foo_s:
				# 	if self.stim_shift < 0:
				# 		stim_hist.append(1)
				# 	else:
				# 		foo2[-1] = 1
				# else:
				# 	if self.stim_shift < 0:
				# 		stim_hist.append(0)
				# 	else:
				# 		foo2[-1] = 0
				# look for if i+self.stim_shift is in foo_s
				if i+self.stim_shift in foo_s:
					foo2[-1] = 1
			ys.append(foo2)

		ys = np.asarray(ys, dtype='uint8').squeeze()
		if self.N == 2:
			try:
				ys = ys.reshape(ys.shape[0], 1)
			except:
				pass
		# if self.stim_shift > 0:
		# 	stims = ys[:, -1]
		# 	stims = stims[self.stim_shift:]
		# 	ys = ys[:-self.stim_shift, :]
		# 	ys[:, -1] = stims

		# counter = np.sum(ys, axis=0)
		# for i in range(ys.shape[
		# 	               1] - 1):  # I believe we will not want to remove the stimulus channel even if it less than threshold but if not remove -1
		# 	if counter[i] < self.threshold:
		# 		ys[:, i] = 0

		return ys

	def GLM(self, dat, num_split=0, dt=0.005):
		# dt chosen as maximum reasonable from Rhea's calculation (for computational efficiency)
		# number of spikes calculation with the Rhea-found dt
		# unit conversion, check this is right
		if self.data_type=='new':
			Ts = dat['data']['Ts'][0][0]/800.0
		elif self.data_type=='old':
			Ts = dat['data']['Ts'][0][0]/0.05
		#
		Cs = dat['data']['Cs'][0][0]
		Stims = dat['data']['StimTimes'][0][0]
		#
		tmin = num_split*np.min([np.max(Ts),np.max(Stims)])/self.splits
		tmax = (num_split+1)*np.min([np.max(Ts),np.max(Stims)])/self.splits
		num_neurons = np.max(Cs)
		num_inputs = num_neurons+1 # can change this
		num_timepoints = int((tmax-tmin)/dt)
		# Note: we are now making models based on data where there is no stimulation.
		# Could break up the data into stim and no stim period.
		X = np.zeros([num_timepoints,num_inputs])
		# second 60 dimensions of X[i,:] are how many times neurons have fired previously
		# first dimension is the time since last stimulus
		N = np.zeros([num_neurons,num_timepoints])
		for i in range(num_timepoints):
			# figure out which times are in that time range
			mask = (Ts<tmin+(i+1)*dt)*(Ts>=tmin+i*dt)
			neurs = Cs[mask]
			for j in np.unique(neurs):
				N[j-1,i] = np.sum(neurs==j) # number of times you see j
			# now to get the input
			# first, the previous stimulus time since last stimulus
			mask = Stims<=i*dt
			if np.sum(mask)==0:
				last_stim_time = -1000/dt
			else:
				last_stim_time = np.max(Stims[mask])
			X[i,0] = i*dt-last_stim_time # should we make it exp(-X[i,0]/tau) and learn tau?  Still convex, I think.
			# second, the previous neural firing
			if i>0:
				X[i,1:] = N[:,i-1]
		# get rid of inactive neurons
		mask = np.sum(N,axis=1)>0
		num_neurons = np.sum(mask)
		num_inputs = num_neurons+2
		X = X[:,np.hstack([True,mask])]
		N = N[mask]
		X = np.hstack([X,np.ones([num_timepoints,1])])
		# divide the memories in half, one to train and one to test, with data from each split
		train_inds = []
		test_inds = []
		all_inds = np.arange(num_timepoints)
		foo = np.random.choice(a=all_inds,size=int(len(all_inds)/2),replace=False)
		train_inds.append(foo)
		test_inds.append(np.setdiff1d(all_inds,foo))
		# train GLM with Rhea-found functional forms on train data
		#
		def f(x):
			return (x>0)*x+1e-5
		def fprime(x):
			return (x>0)+0.0
		def MAP(k,X,n,s=0.1):
			logP = -np.inner(k,k)/2/s**2
			h = np.dot(X,k)
			logP += np.inner(n,np.log(f(h)))-dt*np.sum(f(h))
			return logP
		def gradMAP(k,X,n,s=0.1):
			grad = -k/s**2
			h = np.dot(X,k)
			foo = fprime(h)/f(h)
			grad += np.dot(n*foo,X)
			foo = fprime(h)
			grad -= np.dot(foo,X)*dt
			return grad
		rec_field = {}
		logL = 0
		for i in range(num_neurons):
			# put in correct s from Rhea
			foo = spo.minimize(lambda x: -MAP(x,X[train_inds,:],N[i,train_inds],s=0.01),
				x0=0.01*np.ones(num_inputs+1),
				jac=lambda x: -gradMAP(x,X[train_inds,:],N[i,train_inds],s=0.01))
			rec_field[str(i)] = foo.x
		# get accuracy on test data
		# first get statistics about p(xpast,xfut)
		num = 100
		p_joint = np.zeros([2,num+1]) # can choose something larger than 100
		pasts = np.linspace(0,num*dt,num+1)
		deltat = self.stim_shift*dt
		for i in range(num_timepoints):
			mask = Stims<=i*dt
			if np.sum(mask)==0:
				last_stim_time = -1000/dt
			else:
				last_stim_time = np.max(Stims[mask])
			past_stim = i*dt-last_stim_time
			#
			mask = (Stims>=i*dt+deltat)*(Stims<(i+1)*dt+deltat)
			future_stim = np.min([1,np.sum(mask)])
			# add to statistics
			past_ind = np.min([num,int(past_stim/dt)]) # ignore differences above 100*dt
			p_joint[int(future_stim),past_ind] += 1
		# the alphabet from which future stimulus is drawn
		xs = [0,1]
		# get statistics on p(X_fut,n)
		foo = np.unique(N[:,test_inds],axis=1)
		xfuts = {}
		for ns in foo:
			# find all timepoints with that neural activity
			inds = []
			for ind in test_inds:
				#range(num_timepoints):
				if Ns[:,ind]==ns:
					inds.append(ind)
			# find xfut for all these timepoints
			xfuts_true = []
			for i in xrange(1,10): #????
				if i in inda:
					pass
				mask = (Stims>=i*dt+deltat)*(Stims<(i+1)*dt+deltat)
				future_stim = np.min([1,np.sum(mask)])
				xfuts_true.append(future_stim)
			# find posterior accordingly and find prediction
			posterior = np.zeros(len(xs))
			for i in inds:
				likelihood = 0
				for j in range(num_neurons):
					likelihood += np.exp(N[j,i]*np.log(f(np.dot(rec_field[str(j)],X[i,:])))-f(np.dot(rec_field[str(j)],X[i,:]))*dt)
				posterior += likelihood*p_joint[:,X[i,0]] # \sum_{xpast} p(xfut,xpast)*p(n|xpast) = p(x_fut,n)
			xhat = xs[np.argmax(posterior)]
			xfuts[str(ns)] = [xhat, xfuts_true]
		# use this to compute accuracy score
		# get the probability of each neural configuration
		prob_ns = {}
		for ind in test_inds:
			ns = N[:,ind]
			if str(ns) in prob_ns.keys():
				prob_ns[str(ns)] = 1
			else:
				prob_ns[str(ns)] += 1
		accuracy = 0
		for key in xfuts.keys():
			xhat = xfuts[key][0]
			xfuts_true = xfuts[key][1]
			accuracy_given_n = np.sum([xhat==xfuts_true[i] for i in len(xfuts_true)])/len(xfuts_true)
			accuracy += prob_ns[key]*accuracy_given_n/len(test_inds)
		accuracy /= len(xfuts.keys())
		return accuracy

	def clean_Cs_and_Ts(self, Cs, Ts, threshold=80_000, last_index=0):
		if 60 not in list(Cs):
			return np.array(Cs), np.array(Ts)
		first_marker = 0
		counter = 0
		Cs_beginning = list(Cs[:last_index])
		Ts_beginning = list(Ts[:last_index])
		Cs = list(Cs[last_index:])
		Ts = list(Ts[last_index:])
		index1 = 0
		index2 = 0
		for k, neuron in enumerate(Cs):
			if (neuron == 60) & (first_marker == 0):
				index1 = k
				first_marker = 1
				continue
			elif neuron == 60:
				index2 = k
				counter = 0
			elif first_marker == 1:
				counter += 1
			if (counter > threshold) or ((k + 1) == len(Cs)):
				cutout = list(range(index1, index2 + 1))
				Cs = [b for a, b in enumerate(Cs) if a not in cutout]
				Cs = Cs_beginning + Cs
				Ts = [b for a, b in enumerate(Ts) if a not in cutout]
				Ts = Ts_beginning + Ts
				return self.clean_Cs_and_Ts(Cs, Ts, threshold, index1 + len(Cs_beginning) + 2 * threshold)

	def run_multiprocessing_J(self, filename):
		dat = spio.loadmat(os.path.join(self.in_directory, filename))
		ys = self.binaryVecs(dat, dt=self.dt)
		self.experiments.append(ys)
		self.filenames.append(filename)
		y_sparse = csr_matrix(ys, dtype='uint8')
		filename = filename[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
		save_npz(os.path.join(self.out_directory, filename), y_sparse)

	def mutInfo_NSB(self, xs, ys, Kx, Ky):
		# use NSB entropy estimator
		# first get nXY and nX and nY
		# could probably just use np.histogram
		nX = {}
		for x in xs:
			if str(x) in nX:
				nX[str(x)] += 1
			else:
				nX[str(x)] = 1

		nY = {}
		for y in ys:
			if str(y) in nY:
				nY[str(y)] += 1
			else:
				nY[str(y)] = 1

		nXY = {}
		for i in range(len(xs)):
			x = xs[i]
			y = ys[i]
			if str(x) + '+' + str(y) in nXY:
				nXY[str(x) + '+' + str(y)] += 1
			else:
				nXY[str(x) + '+' + str(y)] = 1

		nX = np.asarray([nx for nx in nX.values()])
		nY = np.asarray([ny for ny in nY.values()])
		nXY = np.asarray([nxy for nxy in nXY.values()])
		#
		Kxy = Kx * Ky

		#
		# now use the following defn
		def entropy_NSB(ns, K):
			ns = ns[ns > 0]
			N = np.sum(ns)

			def Lagrangian(beta):
				K0 = K - len(ns)
				L = -np.sum(sps.gammaln(beta + ns)) - K0 * sps.gammaln(beta) + K * sps.gammaln(beta) - sps.gammaln(
					K * beta) + sps.gammaln(K * beta + N)
				return L

			# Before: find the beta that minimizes L
			ans = spo.minimize_scalar(lambda x: Lagrangian(x), bounds=[(0, None)])
			b = ans.x
			# calculate average S
			foos = (ns + b) * (sps.psi(N + K * b + 1) - sps.psi(ns + b + 1)) / (N + K * b)
			K0 = K - len(ns)
			S = np.sum(foos) + (K0 * b * (sps.psi(N + K * b + 1) - sps.psi(b + 1)) / (N + K * b))

			def avgS2(ns, K, b):
				N = np.sum(ns)
				K0 = K - len(ns)
				# calculate T
				foo1 = (sps.psi(ns + b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1,
				                                                                           ns + b + 2) - sps.polygamma(
					1, N + K * b + 2)
				T = np.sum((ns + b) * (ns + b + 1) * foo1 / (N + K * b) / (N + K * b + 1))
				foo1 = (sps.psi(b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1, b + 2) - sps.polygamma(1,
				                                                                                                N + K * b + 2)
				T += K0 * b * (b + 1) * foo1 / (N + K * b) / (N + K * b + 1)

				# calculate R
				def r(ni, nj, N, K, b):
					alphai = ni + b
					alphaj = nj + b
					foo1 = (sps.psi(alphai) - sps.psi(N + K * b + 1)) * (
							sps.psi(alphaj) - sps.psi(N + K * b + 1)) - sps.polygamma(1, N + K * b + 2)
					foo1 *= alphaj * alphai / (N + K * b) / (N + K * b + 1)
					return foo1

				foo1 = (ns + b) * (sps.psi(ns + b) - sps.psi(N + K * b + 1))
				R = (np.sum(np.outer(foo1, foo1)) - np.sum(np.outer(ns + b, ns + b)) * sps.polygamma(1,
				                                                                                     N + K * b + 2)) / (
						    N + K * b) / (N + K * b + 1)
				R -= np.sum(r(ns, ns, N, K, b))
				R += K0 * np.sum(r(ns, 0, N, K, b) + r(0, ns, N, K, b))
				if K0 > 0:
					R += np.exp(np.log(K0) + np.log(K0 - 1) + np.log(r(0, 0, N, K, b)))
				return R + T

			S2 = avgS2(ns, K, b)
			return S, S2 - S ** 2

		#
		SXY, varSXY = entropy_NSB(nXY, Kxy)
		SX, varSX = entropy_NSB(nX, Kx)
		SY, varSY = entropy_NSB(nY, Ky)
		return SX + SY - SXY, np.sqrt(varSXY + varSX + varSY)

	# figure out which neuron to focus on
	def MI_subset(self, xs, ys, maxChange=0.01, maxNeurons=7):
		# get the best neuron first
		mis = []
		var_mis = []
		for n in range(self.N-1):
			foo_y = ys[:, n]
			foo_y = [[y] for y in foo_y]
			foo = self.mutInfo_NSB(xs, foo_y, 2, 2)
			mis.append(foo[0])
			var_mis.append(foo[1])
		MI = [np.max(mis)]
		var_MI = [var_mis[np.argmax(mis)]]
		best_neurons = [np.argmax(mis)]
		#
		deltaMI = np.inf
		k = 1
		# len(best_neurons)
		while (np.abs(deltaMI) > maxChange) & (len(best_neurons) < maxNeurons):
			# choose the next neuron to add
			mis = []
			var_mis = []
			for j in range(self.N-1):
				if (j in best_neurons) & (self.N > 2):
					mis.append(0)
					var_mis.append(0)
				else:
					inds = np.hstack([best_neurons, j])
					foo_y = ys[:, inds]
					foo = list(self.mutInfo_NSB(xs, foo_y, 2, np.power(2, k + 1)))
					mis.append(foo[0])
					var_mis.append(foo[1])
			MI.append(np.max(mis))
			if MI[-2] != 0:
				deltaMI = (MI[-1] - MI[-2]) / MI[-2]
			else:
				deltaMI = (MI[-1] - MI[-2] / 1e-8)
			var_MI.append(var_mis[np.argmax(mis)])
			best_neurons = np.hstack([best_neurons, np.argmax(mis)])
			k += 1
		return MI[-1], var_MI[-1], best_neurons

	def get_dats(self, dts=None, get_N=False):
		self.dats = []
		self.filenames = []
		if dts is not None:
			if type(dts) != list:
				dts = [dts]
			for dt in dts:
				dats = []
				filenames = []
				self.dt = dt
				input_files = np.asarray(os.listdir(self.in_directory))
				print("The following files are in your input directory: \n")
				if self.requested_files is None:
					for k, file in enumerate(input_files):
						print(f'File {k}: {file}')
					requested_files = input('Please enter the index of files your would like to analyze separated by a comma. \n '
					                        'For example, if you wanted the first and third file, type "0, 2" \n '
					                        'If you would like to use all listed files, just press Enter')
					if requested_files == '':
						requested_files = input_files
					else:
						requested_files = requested_files.split(',')
						for k, file in enumerate(requested_files):
							requested_files[k] = int(file)
						requested_files = input_files[requested_files]
					self.requested_files = requested_files
				for file in self.requested_files:
					filename = file[:-4] + f'_N_{self.N}_{self.dt}_sparse.npz'
					if filename not in os.listdir(self.out_directory):
						print(f'---------------- The file {filename} was not found in the out directory  -----------------')
						print(f'---------------------- Importing .mat file: {file} instead ----------------------')
						dats.append(spio.loadmat(os.path.join(self.in_directory, file)))
						filenames.append(os.path.join(self.out_directory, filename))
					else:
						print(f'------------------ Loading file: {filename} from out directory --------------------')
						dats.append(None)
						filenames.append(os.path.join(self.out_directory, filename))
				self.dats.append(dats)
				self.filenames.append(filenames)
		else:
			dts = self.dt
			return self.get_dats(dts)
		return self.dats, self.filenames

	def get_MI_estimates(self, dts=np.asarray([0.01, 0.03, 0.1, 0.3, 1])):
		dats, filenames = self.get_dats(dts)
		allMIs = []
		allstdMIs = []
		MIs = np.zeros([len(dts), len(dats[0]), self.splits])
		stdMIs = np.zeros([len(dts), len(dats[0]), self.splits])
		for i in range(len(dts)):
			dat_ys = []
			dat = dats[i]
			for k in range(len(dat)):
				# for k, dat in enumerate(dats):
				if dat[k] is None:
					try:
						ys = load_npz(filenames[i][k]).toarray()
					except:
						file = filenames[i][k][len(str(self.out_directory)):-15-len(str(self.N))-len(str(self.dt))] + '.mat'
						dat1 = spio.loadmat(os.path.join(self.in_directory, file))
						ys = self.binaryVecs(dat1, dt=dts[i])
						y_sparse = csr_matrix(ys, dtype='uint8')
						save_npz(filenames[i][k], y_sparse)
				else:
					ys = self.binaryVecs(dat[k], dt=dts[i])
					y_sparse = csr_matrix(ys, dtype='uint8')
					save_npz(filenames[i][k], y_sparse)

				dat_ys.append(ys)
				xs = ys[:, :self.N-1]
				ys = ys[:, -1]
				xs_chunked = self.chunked(xs, self.splits)
				ys_chunked = self.chunked(ys, self.splits)
				MI_chunks = []
				varMI_chunks = []
				for xs_chunk, ys_chunk in zip(xs_chunked, ys_chunked):
					MI, varMI, best_neurons = self.MI_subset(ys_chunk, xs_chunk, 0.05)
					MI_chunks.append(MI)
					varMI_chunks.append(varMI)
				MIs[i, k, :] = np.array(MI_chunks).squeeze()
				stdMIs[i, k, :] = np.sqrt(np.array(varMI_chunks).squeeze())
			self.experiments.append(dat_ys)
			np.savez(filenames[i][k][:-4]+'_MI_data.npz', MIs=MIs, stdMIs=stdMIs, dt=dts[i])
			# allMIs.append(MIs)
			# allstdMIs.append(stdMIs)
		# allMIs = np.array(allMIs).squeeze()
		# allstdMIs = np.array(allstdMIs).squeeze()
		allMIs = np.array(MIs).squeeze()
		allstdMIs = np.array(stdMIs).squeeze()
		return allMIs, allstdMIs

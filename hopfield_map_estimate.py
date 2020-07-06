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

# TODO: Make the mutual information estimate save/load data for faster processing time
# TODO: Select N based on data
# TODO make data_type auto select, so that a folder can have multiple data types

class ModifiedHopfieldNet:
	"""
		Arguments:
		N = number of nodes to build Hopfield network with
		in_directory = the relative path to a folder containing the raw .mat files
		out_directory = the relative path to a folder that you want to store the python data. If not used, defaults to
						in_directory
		splits: the number of times to split the data to train on each portion (slices sequentially
			--> default: 3 splits = beginning, middle, end thirds of experiment)
		train_percent = the percentage of each chunk of data (the number of chunks as defined by splits) that will be
			used to train a Hopfield network --> default: 0.66
		num_nets: The number of replications to use when training Hopfield networks --> default: 5
		exp_type = 'map' to conduct a MAP estimate analysis, J to analyze the changing connectivity matrices through time.
			--> default: 'J'
	"""
	def __init__(self, in_directory, out_directory=None, splits=3, train_percent=0.66, num_nets=5, exp_type='J',
														data_type='old', dt=800, n_jobs=1):
		self.in_directory = in_directory
		self.type = exp_type
		if out_directory is None:
			self.out_directory = self.in_directory
		else:
			self.out_directory = out_directory
		self.N = self._get_n()
		self.splits = splits
		self.experiments = []
		self.train_percent = train_percent
		self.num_nets = num_nets
		self.networks = []
		self.data_type = data_type
		self.dt = dt
		self.n_jobs = n_jobs
		self.filenames = []
		if (self.type != 'MI') & (self.n_jobs == 1):
			self.load_and_save_data(dt=self.dt)
		if (self.n_jobs > 1) & (self.type == 'J'):
			files = []
			for file in os.listdir(self.in_directory):
				filename = file[:-4] + f'_N_{self.N}_sparse.npz'
				if filename not in os.listdir(self.out_directory):
					files.append(file)
			p = Pool(self.n_jobs)
			p.map(self.run_multiprocessing_j, files)
			p.close()
			p.join()

	def _get_n(self):
		# dats = self.get_dats()
		# try:
		# 	Cs = dats[0][0]['Cs']
		# except:
		# 	Cs = dats[0][0]['data']['Cs']
		# 	Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
		# N = len(np.unique(Cs))
		# N = max(60, N)
		if self.type == 'J' or self.type == 'MI':
			n = 60
		else:
			n = 61
		return n

	def load_and_save_data(self, **kwargs):
		for file in os.listdir(self.in_directory):
			filename = file[:-4] + f'_N_{self.N}_sparse.npz'
			if filename not in os.listdir(self.out_directory):
				print(f'---------------------- Importing .mat file: {file} ----------------------')
				dat = spio.loadmat(os.path.join(self.in_directory, file))
				ys = self.binary_vectors(dat, **kwargs)
				self.experiments.append(ys)
				y_sparse = csr_matrix(ys, dtype='uint8')
				save_npz(os.path.join(self.out_directory, filename), y_sparse)
			else:
				print(f'---------------------- Loading file: {filename} ----------------------')
				ys = load_npz(os.path.join(self.out_directory, filename)).toarray()
				self.experiments.append(ys)

	def build_and_train_networks(self):
		for i, memories in enumerate(self.experiments):
			print(f'---------------------- Conducting experiment: {i} ----------------------')
			memories_chunked = self.chunked(memories, self.splits)
			experiment_nets = []
			for j, memory_chunk in enumerate(memories_chunked):
				avg_acc = []
				chunked_nets = []
				for _ in range(self.num_nets):
					hop = hdn.HopfieldNetMPF(N=self.N)
					rand_memories = np.array(
						[random.choice(memory_chunk) for _ in range(round(len(memory_chunk) * self.train_percent))])
					hop.store_patterns_using_mpf(rand_memories)
					if self.type == 'map':
						avg_acc.append(self.get_accuracy(memory_chunk, hop))
					chunked_nets.append(hop)
				experiment_nets.append(chunked_nets)
				if self.type == 'map':
					print(f'Experiment: {i} // Chunk: {j} // Avg Accuracy: {round(np.mean(avg_acc).item(), 3)} +/- '
											f'{round(np.std(avg_acc).item(), 3)}')
				else:
					print(f'Experiment: {i} // Chunk: {j}')
			print(f'---------------------- Finished experiment: {i} ----------------------')
			self.networks.append(experiment_nets)

	@staticmethod
	def chunked(iterable, n):
		chunk_size = int(math.ceil(len(iterable) / n))
		return (iterable[i * chunk_size:i * chunk_size + chunk_size] for i in range(n))

	@staticmethod
	def _calc_probabilities(x, h1, a):
		prob = np.exp(np.dot(h1.T, x) + a)
		if prob > 1:
			return 1
		else:
			return 0

	def get_predictions(self, y, hop):
		j = hop.J
		h = -hop.theta
		a = j[-1, -1]
		b = h[-1]
		# J0 = J[:-1, :-1]
		j = j[-1, :-1]
		# jT = J[-1, :-1]
		# J = J0
		h1 = 2 * j
		# h0 = h
		a += b
		x = y[:-1]
		p = self._calc_probabilities(x, h1, a)
		return p

	def get_accuracy(self, memories, hop):
		accuracy = 0
		y_preds = []
		y_true = []
		for k, i in enumerate(memories):
			y_preds.append(self.get_predictions(i, hop))
			y_true.append(i[-1])
			if y_preds[k] == y_true[k]:
				accuracy += 1
		accuracy = accuracy / len(memories)
		return round(accuracy * 100, 3)

	def get_js(self, filename='Js_Joost.pkl'):
		js = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for network in memory_chunk_networks:
					chunk_nets.append(network._J)
				experiment_nets.append(chunk_nets)
			js.append(experiment_nets)
		js = np.array(js).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(js, file)
		return js

	def get_thetas(self, filename='Thetas_Joost.pkl'):
		thetas = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for network in memory_chunk_networks:
					chunk_nets.append(network._theta)
				experiment_nets.append(chunk_nets)
			thetas.append(experiment_nets)
		thetas = np.array(thetas).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(thetas, file)
		return thetas

	def binary_vectors(self, dat, dt=None):
		if (self.type == 'map') or (self.type == 'MI') & (self.data_type == 'old'):
			if dt is None:
				dt = 0.05
			stimulus_times = dat['StimTimes']
			firing_neurons = np.array(dat['Cs'], dtype='uint32')
			firing_times = np.array(dat['Ts'], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in stimulus_times])
			max_time = np.max([np.max(stimulus_times), np.max(firing_times)])
		elif (self.type == 'J') & (self.data_type == 'old'):
			if dt is None:
				dt = 800
			firing_neurons = np.array(dat['Cs'], dtype='uint32')
			firing_times = np.array(dat['Ts'], dtype='uint32')
			firing_neurons, firing_times = self.clean_data(firing_neurons, firing_times)
			max_time = np.max(firing_times)
		else:
			if dt is None:
				dt = 800
			firing_neurons = dat['data']['Cs']
			firing_times = dat['data']['Ts']
			firing_neurons = np.array([a[0] for a in firing_neurons.tolist()[0][0].tolist()], dtype='uint8')
			firing_times = np.array([a[0] for a in firing_times.tolist()[0][0].tolist()], dtype='uint32')
			firing_neurons_and_times = self.clean_data(firing_neurons, firing_times)
			firing_neurons = firing_neurons_and_times[0]
			firing_times = firing_neurons_and_times[1]
			max_time = np.max(firing_times)

		foo_x = np.asarray([int(i / 0.1) for i in firing_times])

		ys = []
		for i in range(int(max_time / dt)):
			if i in foo_x:
				# which neurons are firing
				idx = (i * dt < firing_times) * (firing_times < (i + 1) * dt)
				neurons = firing_neurons[idx]
				foo2 = np.zeros(self.N)
				foo2[neurons] = 1
			else:
				foo2 = np.zeros(self.N)
			# is the stimulus firing
			if (self.type == 'map') or (self.type == 'MI'):
				if i in foo_s:
					foo2[-1] = 1
				else:
					foo2[-1] = 0
			ys.append(foo2)

		ys = np.asarray(ys, dtype='uint8').squeeze()
		return ys

	def clean_data(self, firing_neurons, firing_times, threshold=80_000, last_index=0):
		if 60 not in list(firing_neurons):
			return np.array(firing_neurons), np.array(firing_times)
		first_marker = 0
		counter = 0
		beginning_neurons = list(firing_neurons[:last_index])
		beginning_times = list(firing_times[:last_index])
		firing_neurons = list(firing_neurons[last_index:])
		firing_times = list(firing_times[last_index:])
		index1 = 0
		index2 = 0
		for k, neuron in enumerate(firing_neurons):
			if (neuron == 60) & (first_marker == 0):
				index1 = k
				first_marker = 1
				continue
			elif neuron == 60:
				index2 = k
				counter = 0
			elif first_marker == 1:
				counter += 1
			if (counter > threshold) or ((k + 1) == len(firing_neurons)):
				cutout = list(range(index1, index2+1))
				firing_neurons = [b for a, b in enumerate(firing_neurons) if a not in cutout]
				firing_neurons = beginning_neurons + firing_neurons
				firing_times = [b for a, b in enumerate(firing_times) if a not in cutout]
				firing_times = beginning_times + firing_times
				return self.clean_data(firing_neurons, firing_times, threshold, index1 + len(beginning_neurons) + 2 * threshold)

	def run_multiprocessing_j(self, filename):
		dat = spio.loadmat(os.path.join(self.in_directory, filename))
		ys = self.binary_vectors(dat, dt=self.dt)
		self.experiments.append(ys)
		self.filenames.append(filename)
		y_sparse = csr_matrix(ys, dtype='uint8')
		filename = filename[:-4] + f'_N_{self.N}_sparse.npz'
		save_npz(os.path.join(self.out_directory, filename), y_sparse)

	@staticmethod
	def mut_info_nsb(xs, ys, k_x, k_y):
		# use NSB entropy estimator
		# first get n_xy and n_x and n_y
		# could probably just use np.histogram
		n_x = {}
		for x in xs:
			if str(x) in n_x:
				n_x[str(x)] += 1
			else:
				n_x[str(x)] = 1

		n_y = {}
		for y in ys:
			if str(y) in n_y:
				n_y[str(y)] += 1
			else:
				n_y[str(y)] = 1

		n_xy = {}
		for i in range(len(xs)):
			x = xs[i]
			y = ys[i]
			if str(x) + '+' + str(y) in n_xy:
				n_xy[str(x) + '+' + str(y)] += 1
			else:
				n_xy[str(x) + '+' + str(y)] = 1

		n_x = np.asarray([nx for nx in n_x.values()])
		n_y = np.asarray([ny for ny in n_y.values()])
		n_xy = np.asarray([nxy for nxy in n_xy.values()])
		#
		kxy = k_x * k_y

		#
		# now use the following defn
		def entropy_nsb(ns, k):
			ns = ns[ns > 0]
			n = np.sum(ns)

			def lagrangian(beta):
				k0 = k - len(ns)
				lag = -np.sum(sps.gammaln(beta + ns)) - k0 * sps.gammaln(beta) + k * sps.gammaln(beta) - sps.gammaln(
					k * beta) + sps.gammaln(k * beta + n)
				return lag

			# Before: find the beta that minimizes L
			ans = spo.minimize_scalar(lambda x: lagrangian(x), bounds=[(0, None)])
			b = ans.x
			# calculate average S
			foo = (ns + b) * (sps.psi(n + k * b + 1) - sps.psi(ns + b + 1)) / (n + k * b)
			k0 = k - len(ns)
			s = np.sum(foo) + (k0 * b * (sps.psi(n + k * b + 1) - sps.psi(b + 1)) / (n + k * b))

			def avg_s2(ns, k, b):
				n = np.sum(ns)
				k0 = k - len(ns)
				# calculate t
				foo1 = (sps.psi(ns + b + 1) - sps.psi(n + k * b + 1)) ** 2 + sps.polygamma(1, ns + b + 2) - sps.polygamma(
					1, n + k * b + 2)
				t = np.sum((ns + b) * (ns + b + 1) * foo1 / (n + k * b) / (n + k * b + 1))
				foo1 = (sps.psi(b + 1) - sps.psi(n + k * b + 1)) ** 2 + sps.polygamma(1, b + 2) - sps.polygamma(1, n + k * b + 2)
				t += k0 * b * (b + 1) * foo1 / (n + k * b) / (n + k * b + 1)

				# calculate r
				def corr(ni, nj, n, k, b):
					alpha_i = ni + b
					alpha_j = nj + b
					foo1 = (sps.psi(alpha_i) - sps.psi(n + k * b + 1)) * (
								sps.psi(alpha_j) - sps.psi(n + k * b + 1)) - sps.polygamma(1, n + k * b + 2)
					foo1 *= alpha_j * alpha_i / (n + k * b) / (n + k * b + 1)
					return foo1

				foo1 = (ns + b) * (sps.psi(ns + b) - sps.psi(n + k * b + 1))
				r = (np.sum(np.outer(foo1, foo1)) - np.sum(np.outer(ns + b, ns + b)) * sps.polygamma(1, n + k * b + 2)) / (
						n + k * b) / (n + k * b + 1)
				r -= np.sum(corr(ns, ns, n, k, b))
				r += k0 * np.sum(corr(ns, 0, n, k, b) + corr(0, ns, n, k, b))
				if k0 > 0:
					r += np.exp(np.log(k0) + np.log(k0 - 1) + np.log(corr(0, 0, n, k, b)))
				return r + t

			s2 = avg_s2(ns, k, b)
			return s, s2 - s ** 2

		#
		sxy, var_sxy = entropy_nsb(n_xy, kxy)
		sx, var_sx = entropy_nsb(n_x, k_x)
		sy, var_sy = entropy_nsb(n_y, k_y)
		return sx + sy - sxy, np.sqrt(var_sxy + var_sx + var_sy)

	# figure out which neuron to focus on
	def mut_info_subset(self, xs, ys, max_change=0.01):
		# get the best neuron first
		mis = []
		var_mis = []
		for n in range(self.N):
			foo_y = ys[:, n]
			foo_y = [[y] for y in foo_y]
			foo = self.mut_info_nsb(xs, foo_y, 2, 2)
			mis.append(foo[0])
			var_mis.append(foo[1])
		mi = [np.max(mis)]
		var_mi = [var_mis[np.argmax(mis).item()]]
		best_neurons = [np.argmax(mis)]
		#
		delta_mi = np.inf
		k = 1
		#
		while delta_mi > max_change:
			# choose the next neuron to add
			mis = []
			var_mis = []
			for j in range(self.N):
				if j in best_neurons:
					mis.append(0)
					var_mis.append(0)
				else:
					inds = np.hstack([best_neurons, j])
					foo_y = ys[:, inds]
					foo = list(self.mut_info_nsb(xs, foo_y, 2, np.power(2, k + 1)))
					mis.append(foo[0])
					var_mis.append(foo[1])
			mi.append(np.max(mis))
			delta_mi = (mi[-1] - mi[-2]) / mi[-2]
			var_mi.append(var_mis[np.argmax(mis).item()])
			best_neurons = np.hstack([best_neurons, np.argmax(mis)])
			k += 1
		return mi[-1], var_mi[-1], best_neurons

	def get_dats(self, dts=None):
		dats = []
		filenames = []
		for file in os.listdir(self.in_directory):
			dat = spio.loadmat(os.path.join(self.in_directory, file))
			dats.append(dat)
			filename = f'{file} + dts={dts}'
			filenames.append(filename)
		return dats, filenames

	def get_mut_info_estimates(self, dts=np.asarray([0.01, 0.03, 0.1, 0.3, 1])):
		dats, filenames = self.get_dats(dts)
		all_mut_infos = []
		all_std_mut_infos = []
		for k, dat in enumerate(dats):
			mut_infos = np.zeros([3, len(dts), self.splits])
			mut_info_stds = np.zeros([3, len(dts), self.splits])
			dat_ys = []
			for i in range(len(dts)):
				ys = self.binary_vectors(dat, dt=dts[i])
				dat_ys.append(ys)
				xs = ys[:, :self.N]
				ys = ys[:, -1]
				xs_chunked = self.chunked(xs, self.splits)
				ys_chunked = self.chunked(ys, self.splits)
				mut_info_chunks = []
				std_mut_info_chunks = []
				for xs_chunk, ys_chunk in zip(xs_chunked, ys_chunked):
					mut_info, std_mut_info, best_neurons = self.mut_info_subset(ys_chunk, xs_chunk, 0.05)
					mut_info_chunks.append(mut_info)
					std_mut_info_chunks.append(std_mut_info)
				mut_infos[:, i] = np.array(mut_infos).squeeze()
				mut_info_stds[:, i, :] = np.sqrt(np.array(std_mut_info_chunks).squeeze())
			self.experiments.append(dat_ys)
			np.savez(filenames[k], MIs=mut_infos, stdMIs=mut_info_stds, dts=dts)
			all_mut_infos.append(mut_infos)
			all_std_mut_infos.append(mut_info_stds)
		all_mut_infos = np.array(all_mut_infos)
		all_std_mut_infos = np.array(all_std_mut_infos)
		return all_mut_infos, all_std_mut_infos

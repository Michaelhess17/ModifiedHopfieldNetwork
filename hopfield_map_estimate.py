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
import logging
from multiprocessing import Pool

logging.basicConfig(filename='C:/Users/Micha/Desktop/Hopfield Networks/MyLog.log', level=logging.DEBUG)


# TODO: Make the mutual information estimate save/load data for faster processing time
# TODO: Select N based on data
# TODO make data_type auto select, so that a folder can have multiple data types

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
	def __init__(self, in_directory, out_directory=None, splits=3, train_percent=0.66, num_nets=5, exp_type='J',
														data_type='old', dt=800, n_jobs=1):
		self.in_directory = in_directory
		self.type = exp_type
		if out_directory is None:
			self.out_directory = self.in_directory
		else:
			self.out_directory = out_directory
		self.N = self.get_N()
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
			p.map(self.run_multiprocessing_J, files)
			p.close()
			p.join()

	def get_N(self):
		# dats = self.get_dats()
		# try:
		# 	Cs = dats[0][0]['Cs']
		# except:
		# 	Cs = dats[0][0]['data']['Cs']
		# 	Cs = np.array([a[0] for a in Cs.tolist()[0][0].tolist()], dtype='uint8')
		# N = len(np.unique(Cs))
		# N = max(60, N)
		if self.type == 'J' or self.type == 'MI':
			N = 60
		else:
			N = 61
		return N

	def load_and_save_data(self, **kwargs):
		for file in os.listdir(self.in_directory):
			filename = file[:-4] + f'_N_{self.N}_sparse.npz'
			if filename not in os.listdir(self.out_directory):
				print(f'---------------------- Importing .mat file: {file} ----------------------')
				dat = spio.loadmat(os.path.join(self.in_directory, file))
				ys = self.binaryVecs(dat, **kwargs)
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
					print(f'Experiment: {i} // Chunk: {j} // Avg Accuracy: {round(np.mean(avg_acc), 3)} +/- {round(np.std(avg_acc), 3)}')
				else:
					print(f'Experiment: {i} // Chunk: {j}')
			print(f'---------------------- Finished experiment: {i} ----------------------')
			self.networks.append(experiment_nets)

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

	def get_accuracy(self, memories, hop):
		accuracy = 0
		y_preds = []
		y_true = []
		for k, i in enumerate(memories):
			y_preds.append(self.get_preds(i, hop))
			y_true.append(i[-1])
			if y_preds[k] == y_true[k]:
				accuracy += 1
		accuracy = accuracy / len(memories)
		return round(accuracy * 100, 3)

	def get_js(self, filename='Js_Joost.pkl'):
		Js = []
		for experiment_networks in self.networks:
			experiment_nets = []
			for memory_chunk_networks in experiment_networks:
				chunk_nets = []
				for network in memory_chunk_networks:
					chunk_nets.append(network._J)
				experiment_nets.append(chunk_nets)
			Js.append(experiment_nets)
		Js = np.array(Js).squeeze()
		with open(filename, 'wb') as file:
			pickle.dump(Js, file)
		return Js

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

	def binaryVecs(self, dat, dt=None):
		if (self.type == 'map') or (self.type == 'MI') & (self.data_type == 'old'):
			if dt is None:
				dt = 0.05
			StimTimes = dat['StimTimes']
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			foo_s = np.asarray([int(i / dt) for i in StimTimes])
			Tmax = np.max([np.max(StimTimes), np.max(Ts)])
		elif (self.type == 'J') & (self.data_type == 'old'):
			if dt is None:
				dt = 800
			Cs = np.array(dat['Cs'], dtype='uint32')
			Ts = np.array(dat['Ts'], dtype='uint32')
			Cs, Ts = self.clean_Cs_and_Ts(Cs, Ts)
			Tmax = np.max(Ts)
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

		foo_x = np.asarray([int(i / 0.1) for i in Ts])

		ys = []
		for i in range(int(Tmax / dt)):
			if i in foo_x:
				# which neurons are firing
				inds = (i * dt < Ts) * (Ts < (i + 1) * dt)
				neurons = Cs[inds]
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
				cutout = list(range(index1, index2+1))
				Cs = [b for a, b in enumerate(Cs) if a not in cutout]
				Cs = Cs_beginning + Cs
				Ts = [b for a, b in enumerate(Ts) if a not in cutout]
				Ts = Ts_beginning + Ts
				return self.clean_Cs_and_Ts(Cs, Ts, threshold, index1+len(Cs_beginning)+2*threshold)

	def run_multiprocessing_J(self, filename):
		dat = spio.loadmat(os.path.join(self.in_directory, filename))
		ys = self.binaryVecs(dat, dt=self.dt)
		self.experiments.append(ys)
		self.filenames.append(filename)
		y_sparse = csr_matrix(ys, dtype='uint8')
		filename = filename[:-4] + f'_N_{self.N}_sparse.npz'
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
				foo1 = (sps.psi(ns + b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1, ns + b + 2) - sps.polygamma(
					1, N + K * b + 2)
				T = np.sum((ns + b) * (ns + b + 1) * foo1 / (N + K * b) / (N + K * b + 1))
				foo1 = (sps.psi(b + 1) - sps.psi(N + K * b + 1)) ** 2 + sps.polygamma(1, b + 2) - sps.polygamma(1, N + K * b + 2)
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
				R = (np.sum(np.outer(foo1, foo1)) - np.sum(np.outer(ns + b, ns + b)) * sps.polygamma(1, N + K * b + 2)) / (
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
	def MI_subset(self, xs, ys, maxChange=0.01):
		# get the best neuron first
		mis = []
		var_mis = []
		for n in range(self.N):
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
		#
		while deltaMI > maxChange:
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
					foo = list(self.mutInfo_NSB(xs, foo_y, 2, np.power(2, k + 1)))
					mis.append(foo[0])
					var_mis.append(foo[1])
			MI.append(np.max(mis))
			deltaMI = (MI[-1] - MI[-2]) / MI[-2]
			var_MI.append(var_mis[np.argmax(mis)])
			best_neurons = np.hstack([best_neurons, np.argmax(mis)])
			k += 1
		return MI[-1], var_MI[-1], best_neurons

	def get_dats(self, dts=None):
		dats = []
		filenames = []
		for file in os.listdir(self.in_directory):
			dat = spio.loadmat(os.path.join(self.in_directory, file))
			dats.append(dat)
			filename = f'{file} + dts={dts}'
			filenames.append(filename)
		return dats, filenames

	def get_MI_estimates(self, dts=np.asarray([0.01, 0.03, 0.1, 0.3, 1])):
		dats, filenames = self.get_dats(dts)
		allMIs = []
		allstdMIs = []
		for k, dat in enumerate(dats):
			MIs = np.zeros([3, len(dts), self.splits])
			stdMIs = np.zeros([3, len(dts), self.splits])
			dat_ys = []
			for i in range(len(dts)):
				ys = self.binaryVecs(dat, dt=dts[i])
				dat_ys.append(ys)
				xs = ys[:, :self.N]
				ys = ys[:, -1]
				xs_chunked = self.chunked(xs, self.splits)
				ys_chunked = self.chunked(ys, self.splits)
				MI_chunks = []
				varMI_chunks = []
				for xs_chunk, ys_chunk in zip(xs_chunked, ys_chunked):
					MI, varMI, best_neurons = self.MI_subset(ys_chunk, xs_chunk, 0.05)
					MI_chunks.append(MI)
					varMI_chunks.append(varMI)
				MIs[:, i] = np.array(MIs).squeeze()
				stdMIs[:, i, :] = np.sqrt(np.array(varMI_chunks).squeeze())
			self.experiments.append(dat_ys)
			np.savez(filenames[k], MIs=MIs, stdMIs=stdMIs, dts=dts)
			allMIs.append(MIs)
			allstdMIs.append(stdMIs)
		allMIs = np.array(allMIs)
		allstdMIs = np.array(allstdMIs)
		return allMIs, allstdMIs
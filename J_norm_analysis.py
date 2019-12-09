from hopfield_map_estimate import ModifiedHopfieldNet
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import random
import pickle
import os

if __name__ == '__main__':
	net = ModifiedHopfieldNet(in_directory='toy_data2/', out_directory='new_data3/', exp_type='J', splits=1, num_nets=100,
							  train_percent=0.66, data_type='new', n_jobs=1)
	net.build_and_train_networks()


	Js = net.get_js()

	# baseline, baseline carbachol, rat_stim_1elec_1st, rat_stim_2elec_1st, rat_stim_1elec_2nd, newest_data.
	order = [3, 4, 0, 2, 1]
	new_Js = np.zeros_like(Js)
	for k, i in enumerate(order):
		J_placeholder = np.array(Js[i,:,:,:]).squeeze()
		new_Js[k,:,:,:] = J_placeholder
	Js = np.array(new_Js)
	print(Js.shape)

	delta = np.zeros(Js.shape[0]-1)
	J1 = None
	average_Js = []
	for k, J in enumerate(Js):
		print(f'k: {k} // J shape: {J.shape}')
		J = np.mean(np.array(J).squeeze(), axis=0)
		average_Js.append(J)
		fig = sb.heatmap(J, vmin=0, vmax=1.0)
		fig = fig.get_figure()
		plt.show()
		fig.savefig(f'J Matrix {k}')
		if J1 is None:
			J1 = J
			continue
		delta[k-1] = np.sum(np.abs(J1 - J))

	J2 = average_Js[2]
	J3 = average_Js[3]
	J4 = average_Js[4]
	print(np.sum(np.abs(J2 - J3)))
	print(np.sum(np.abs(J2 - J4)))
	print(np.sum(np.abs(J3 - J4)))

	thetas = net.get_thetas()


	deltas = []
	for k, J in enumerate(Js):
		delta = []
		for i in range(10000):
			if k+1 != len(Js):
				idx1 = random.choice(range(len(J.tolist())))
				idx2 = random.choice(range(len(Js[k+1].tolist())))
				J = Js[:,idx1,:,:].squeeze()
				J1 = Js[:,idx2,:,:].squeeze()
				delta.append(np.sum(np.abs(J1 - J)))
		if k + 1 != len(Js):
			deltas.append(delta)
	deltas = np.array(deltas).squeeze()
	# print(np.mean(deltas, axis=1))
	# print(np.std(deltas, axis=1))

	J2 = Js[2,:,:,:].squeeze()
	J4 = Js[4,:,:,:].squeeze()

	delta = []
	for i in range(10000):
		idx1 = random.choice(range(len(J2.tolist())))
		idx2 = random.choice(range(len(J4.tolist())))
		J = J2[idx1,:,:].squeeze()
		J1 = J4[idx2,:,:].squeeze()
		delta.append(np.sum(np.abs(J1 - J)))
	deltas2 = np.concatenate((deltas, np.array(delta).reshape(1,-1)), axis=0)
	print(np.mean(deltas2, axis=1))
	print(np.std(deltas2, axis=1))

# [647.13074422 661.89628699 656.88877151 651.90698891 295.16562999] means
# [390.05533673 398.9264911  393.80303693 385.2838112  276.88412004] stds


def pkl2np(pickle_path, save_path):
	with open(pickle_path, 'rb') as file:
		data = pickle.load(file)
		np.savez(file=save_path, data=data)


for file in os.listdir('./'):
	if file.endswith(".pkl"):
		pkl2np(file[:-4], file[:-4]+'.npz')

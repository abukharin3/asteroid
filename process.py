import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import pandas as pd


def process_ccsd(molecule_name='benzene', n_pts=200):
	if molecule_name == 'benzene':
		molecule = np.load("raw_data/benzene_ccsd_t/benzene_ccsd_t-train.npz")
		test_molecule = np.load("raw_data/benzene_ccsd_t/benzene_ccsd_t-test.npz")
	elif molecule_name == 'aspirin':
		molecule = np.load("raw_data/aspirin_ccsd/aspirin_ccsd-train.npz")
		test_molecule = np.load("raw_data/aspirin_ccsd/aspirin_ccsd-test.npz")
	elif molecule_name == 'ethanol':
		molecule = np.load("raw_data/ethanol_ccsd_t/ethanol_ccsd_t-train.npz")
		test_molecule = np.load("raw_data/ethanol_ccsd_t/ethanol_ccsd_t-test.npz")
	elif molecule_name == 'malonaldehyde':
		molecule = np.load("raw_data/malonaldehyde_ccsd_t/malonaldehyde_ccsd_t-train.npz")
		test_molecule = np.load("raw_data/malonaldehyde_ccsd_t/malonaldehyde_ccsd_t-test.npz")
	elif molecule_name == 'toluene':
		molecule = np.load("raw_data/toluene_ccsd_t/toluene_ccsd_t-train.npz")
		test_molecule = np.load("raw_data/toluene_ccsd_t/toluene_ccsd_t-test.npz")

	
	N = molecule['R'].shape[1]
	np.random.seed(0)
	idx = np.arange(molecule['R'].shape[0])
	np.random.shuffle(idx)
	
	train, val, test = {}, {}, {}
	
	train['E'] = molecule['E'][idx][:n_pts, 0]
	train['F'] = molecule['F'][idx][:n_pts].reshape(n_pts * N, 3)
	train['R'] = molecule['R'][idx][:n_pts].reshape(n_pts * N, 3)
	train['Z'] = np.tile(molecule['z'], (n_pts, 1)).astype('float64').reshape(n_pts * N)
	train['N'] = (np.ones(n_pts) * N).astype(int)
	print(train['R'].shape, N)
	
	val['E'] = molecule['E'][idx][950:1000, 0]
	val['F'] = molecule['F'][idx][950:1000].reshape(50 * N, 3)
	val['R'] = molecule['R'][idx][950:1000].reshape(50 * N, 3)
	val['Z'] = np.tile(molecule['z'], (50, 1)).astype('float64').reshape(50 * N)
	val['N'] = (np.ones(50) * N).astype(int)
	
	test['E'] = test_molecule['E'][:, 0]
	test['F'] = test_molecule['F'].reshape(test['E'].shape[0] * N, 3)
	test['R'] = test_molecule['R'].reshape(test['E'].shape[0] * N, 3)
	test['Z'] = np.tile(test_molecule['z'], (test['E'].shape[0], 1)).astype('float64').reshape(test['E'].shape[0] * N)
	test['N'] = (np.ones(test['E'].shape[0]) * N).astype(int)
	
	np.savez_compressed("data/{}_train_gem_ccsd_{}.npz".format(molecule_name, n_pts), E=train['E'],
						F=train['F'], R=train['R'], Z=train['Z'], N=train['N'])
	np.savez_compressed("data/{}_val_gem_ccsd_{}.npz".format(molecule_name, n_pts), E=val['E'],
						F=val['F'], R=val['R'], Z=val['Z'], N=val['N'])
	np.savez_compressed("data/{}_test_gem_ccsd_{}.npz".format(molecule_name, n_pts), E=test['E'],
						F=test['F'], R=test['R'], Z=test['Z'], N=test['N'])


def process_gemnet_revised(fname, molecule_name="aspirin", n_pts=1000, seed=0):
	
	# Split data
	molecule = np.load(fname)
	test_idx = pd.read_csv("raw_data/rmd17/splits/index_test_01.csv", header=None).to_numpy()
	mask = np.ones(len(molecule['energies']), bool)
	mask[test_idx] = 0
	
	train = {}
	val = {}
	test = {}

	N = molecule['coords'].shape[1]
	np.random.seed(seed)
	idx = np.arange(mask.shape[0] - test_idx.shape[0])
	np.random.shuffle(idx)

	train['E'] = molecule['energies'][mask][idx][:n_pts].squeeze()
	train['F'] = molecule['forces'][mask][idx][:n_pts].reshape(n_pts * N, 3)
	train['R'] = molecule['coords'][mask][idx][:n_pts].reshape(n_pts * N, 3)
	train['Z'] = np.tile(molecule['nuclear_charges'], (n_pts, 1)).astype('float64').reshape(n_pts * N)
	train['N'] = (np.ones(n_pts) * N).astype(int)

	val['E'] = molecule['energies'][mask][idx][n_pts:n_pts + 100].squeeze()
	val['F'] = molecule['forces'][mask][idx][n_pts:n_pts + 100].reshape(100 * N, 3)
	val['R'] = molecule['coords'][mask][idx][n_pts:n_pts + 100].reshape(100 * N, 3)
	val['Z'] = np.tile(molecule['nuclear_charges'], (100, 1)).astype('float64').reshape(100 * N)
	val['N'] = (np.ones(100) * N).astype(int)

	test['E'] = molecule['energies'][test_idx].squeeze()
	test['F'] = molecule['forces'][test_idx].reshape(1000 * N, 3)
	test['R'] = molecule['coords'][test_idx].reshape(1000 * N, 3)
	test['Z'] = np.tile(molecule['nuclear_charges'], (1000, 1)).astype('float64').reshape(1000 * N)
	test['N'] = (np.ones(1000) * N).astype(int)

	print(molecule_name)
	print(train['E'].shape)
	print(train['F'].shape)
	print(train['R'].shape)
	print(train['Z'].shape)
	print(train['N'].shape)
	print("--------------------")

	print(val['E'].shape)
	print(val['F'].shape)
	print(val['R'].shape)
	print(val['Z'].shape)
	print(val['N'].shape)
	print("--------------------")

	print(test['E'].shape)
	print(test['F'].shape)
	print(test['R'].shape)
	print(test['Z'].shape)
	print(test['N'].shape)

	np.savez_compressed("data/{}_train_gem_revised_{}.npz".format(molecule_name, n_pts), E=train['E'],
						F=train['F'], R=train['R'], Z=train['Z'], N=train['N'])
	np.savez_compressed("data/{}_val_gem_revised.npz".format(molecule_name), E=val['E'],
						F=val['F'], R=val['R'], Z=val['Z'], N=val['N'])
	np.savez_compressed("data/{}_test_gem_revised.npz".format(molecule_name), E=test['E'],
					   F=test['F'], R=test['R'], Z=test['Z'], N=test['N'])



if __name__ == "__main__":
	process_gemnet_revised('raw_data/rmd17/npz_data/rmd17_aspirin.npz', molecule_name='aspirin', n_pts=2000)
	process_ccsd(molecule_name='aspirin', n_pts=200)
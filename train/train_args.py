#!/usr/bin/env bash
import numpy as np
import preprocess_args as args

# Dataset Dir and params settings.
transfer_flag = False
dipole_mul = False
force_posc = False

dataset_dir = 'QM9_datasets/'
# task = 'Force'
task = 'Dipole'
# task = 'Energy'

# dataset_dir = 'SN2_datasets/'
# task = 'SN2_all'


if task == 'Force':
	dataset = 'QM9under14atoms_force_and_AE_eV'
elif task == 'Energy' or task == 'Dipole':
	dataset = 'QM9under14atoms_dipole_and_AE_eV'
elif task == 'SN2_all':
	dataset = 'SN2_all_potential_energy_eV'
else:
	print('Task not recognized')


# Basis set.
basis_set = '6-311G'  # '6-311G.gbs'

# Setting of a neural network architecture.
dim = 300

n_step = (args.radius_max -
          args.radius_min) / args.radius_step + 1

seq_len = int(((args.n_points - 2) *
               args.n_theta + 2) * n_step)  # atom field number

# Operation for final layer.
operation = 'none' if task == 'Force' or task == 'Dipole' else 'sum'

# Setting of optimization.
batch_size = 4
lr = 1e-3
iteration = 250
dropout = 0

# num_workers=0
num_workers = 0

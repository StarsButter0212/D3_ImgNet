#!/usr/bin/env bash
import preprocess_args as args

# Dataset.
# dataset = 'QM9under7atoms_atomizationenergy_eV'
dataset = 'QM9under14atoms_atomizationenergy_eV'
# dataset = 'QM9over15atoms_atomizationenergy_eV'

# Basis set.
basis_set = '6-311G'  # '6-311G.gbs'

# Setting of a neural network architecture.
dim = 300

n_step = (args.radius_max -
          args.radius_min) / args.radius_step + 1

seq_len = int(((args.n_points - 2) *
               args.n_theta + 2) * n_step)  # atom field number

# Operation for final layer.
operation = 'sum'  # For energy.

# Setting of optimization.
batch_size = 4
lr = 1e-3
iteration = 250
dropout = 0

# num_workers=0
num_workers = 0

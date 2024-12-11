#!/usr/bin/env bash
import numpy as np

# Dataset Dir and params settings.
dataset_dir = 'QM9_datasets/'
task = 'Force'
# task = 'Dipole & Energy'


if task == 'Force':
	dataset_trained = 'QM9under14atoms_force_and_AE_eV'
	dataset_predict = 'QM9over15atoms_force_and_AE_eV'  # Extrapolation.
elif task == 'Dipole & Energy':
	dataset_trained = 'QM9under14atoms_dipole_and_AE_eV'
	dataset_predict = 'QM9over15atoms_dipole_and_AE_eV'
else:
	print('Task not recognized')


# Basis set.
basis_set = '6-311G'

# Grid field.
radius_min = 0.3
radius_max = 0.4
radius_step = 0.1
n_points = 14                       # one circle's points
n_theta = 4
rot_angle = np.pi/n_theta
rot_axis = [0, 0, 1]               # z-axis




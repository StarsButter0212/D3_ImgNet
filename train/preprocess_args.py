#!/usr/bin/env bash
import numpy as np

# Dataset Dir and params settings.
dataset_dir = 'QM9_datasets/'
# task = 'Force'
task = 'Dipole & Energy'

# dataset_dir = 'SN2_datasets/'
# task = 'SN2_all'

if task == 'Force':
	dataset = 'QM9under14atoms_force_and_AE_eV'
elif task == 'Dipole & Energy':
	dataset = 'QM9under14atoms_dipole_and_AE_eV'
elif task == 'SN2_all':
	dataset = 'SN2_all_potential_energy_eV'
else:
	print('Task not recognized')

# Basis set.
basis_set = '6-311G'                # '6-311G.gbs'

# Grid field.
radius_min = 0.3
radius_max = 0.4
radius_step = 0.1
n_points = 14                       # one circle's points
n_theta = 4
rot_angle = np.pi/n_theta
rot_axis = [0, 0, 1]               # z-axis

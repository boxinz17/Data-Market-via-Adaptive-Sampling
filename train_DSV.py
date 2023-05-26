import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
from util import *
import sys

# read the input random seed index
i_rand = int(sys.argv[1])

# the name of dataset and the number of providers
dataset_name = 'CIFAR10'  # acceptable variables are ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10']
n_providers = 100

print('dataset: {}'.format(dataset_name))
print('n_providers: {}'.format(n_providers))

# load the random seed array
with open('rd_seed_array.pickle', 'rb') as f:
    rd_seed_array = pickle.load(f)

# load the data
with open('data/' + dataset_name.lower() + '/corrupted_data_ndevices=' + str(n_providers) + '.pickle', 'rb') as f:
    providers_train_list, val_loader, test_loader = pickle.load(f)

step_size_local = 0.1  # step size for local updates
n_epoch = 1  # number of epochs for local computation of each round

c_tol = 0.01  # tolerance to decide if DSV has converged

if dataset_name == 'CIFAR10':
    max_n_perm = 1000  # maximum number of permutations
    back_check = 30  # the gap to check convergence of DSV
else:
    max_n_perm = 500  # maximum number of permutations
    back_check = 10  # the gap to check convergence of DSV

# set up random seed
print("random seed: {}".format(rd_seed_array[i_rand]))
np.random.seed(rd_seed_array[i_rand])

DSV_list, DSV_convg_measure, DSV, time_used = train_DSV(providers_train_list, val_loader, dataset_name, step_size_local, n_epoch, back_check, c_tol, max_n_perm)

print("Time used: {}(s)".format(time_used))

# store the result
with open('result/' + dataset_name + '/FedAvg/DSV/time_cost_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(time_used, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV_list_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV_list, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV_convg_measure_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV_convg_measure, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV, f)
 
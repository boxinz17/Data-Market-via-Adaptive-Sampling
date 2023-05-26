import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import sys

sys.path.append('../')
from util import *


# read the input random seed index
i_rand = int(sys.argv[1])

# the name of dataset and the number of providers
dataset_name = 'MNIST'  # acceptable variables are ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10']
n_providers = 50  # we choose n_providers=50, 100, 200, 400 in the paper

print('dataset: {}'.format(dataset_name))
print('n_providers: {}'.format(n_providers))

# load the random seed array
with open('../rd_seed_array.pickle', 'rb') as f:
    rd_seed_array = pickle.load(f)

# load the data
with open('data/' + dataset_name.lower() + '/corrupted_data_ndevices=' + str(n_providers) + '.pickle', 'rb') as f:
    providers_train_list, val_loader, test_loader = pickle.load(f)

step_size_local = 0.1  # step size for local updates
step_size_global = step_size_local  # step size for global updates
n_epoch = 1  # number of epochs for local computation of each round
batch_size = int(0.1 * n_providers)  # number of providers chosen in each communication round
n_commun = 500  # number of communications
c_tol = 0.05  # convgernece tolerance to decide if DSV has converged
back_check = 10  # the gap to check convergence of DSV

# set up random seed
print("random seed: {}".format(rd_seed_array[i_rand]))
np.random.seed(rd_seed_array[i_rand])

model, n_access, util_list, test_accu_list, time_used, FedDSV = train_FedAvg_FedDSV(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun, c_tol, back_check)

print("Time used: {}(s)".format(time_used))

# store the result
with open('result/' + dataset_name + '/FedAvg/FedDSV/model_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('result/' + dataset_name + '/FedAvg/FedDSV/n_access_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(n_access, f)

with open('result/' + dataset_name + '/FedAvg/FedDSV/util_list_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(util_list, f)

with open('result/' + dataset_name + '/FedAvg/FedDSV/time_cost_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(time_used, f)

with open('result/' + dataset_name + '/FedAvg/FedDSV/test_accu_list_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(test_accu_list, f)

with open('result/' + dataset_name + '/FedAvg/FedDSV/FedDSV_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(FedDSV, f)
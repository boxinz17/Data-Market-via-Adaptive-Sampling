import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from util import *
import sys

# read the input random seed index
i_rand = int(sys.argv[1])

# the name of dataset and the number of providers
dataset_name = 'CIFAR10' # acceptable variables are ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10']
n_providers = 100

print('dataset: {}'.format(dataset_name))
print('n_providers: {}'.format(n_providers))

# load the random seed array
with open('rd_seed_array.pickle', 'rb') as f:
    rd_seed_array = pickle.load(f)

# load the data
with open('data/' + dataset_name.lower() + '/corrupted_data_ndevices=' + str(n_providers) + '.pickle', 'rb') as f:
    providers_train_list, val_loader, test_loader = pickle.load(f)

step_size_local = 0.1
step_size_global = step_size_local  # step size for global updates
n_epoch = 1  # number of epochs for local computation of each round
batch_size = int(0.1 * n_providers)  # number of providers chosen in each communication round

if dataset_name == 'CIFAR10':
    n_commun = 1000  # number of communications
else:
    n_commun = 500  # number of communications

alpha = 0.01  # parameter of OSMD
learning_rate = 1.0  # parameter of OSMD
print('alpha: {} | learning rate: {}'.format(alpha, learning_rate))

# set up random seed
print("random seed: {}".format(rd_seed_array[i_rand]))
np.random.seed(rd_seed_array[i_rand])

model, n_access, util_list, test_accu_list, time_used, prob_sampling = train_FedAvg_OSMD(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun, alpha, learning_rate)

print("Time used: {}(s)".format(time_used))

# store the result
with open('result/' + dataset_name + '/FedAvg/OSMD/model_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('result/' + dataset_name + '/FedAvg/OSMD/n_access_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(n_access, f)

with open('result/' + dataset_name + '/FedAvg/OSMD/prob_sampling_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(prob_sampling, f)

with open('result/' + dataset_name + '/FedAvg/OSMD/util_list_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(util_list, f)

with open('result/' + dataset_name + '/FedAvg/OSMD/time_cost_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(time_used, f)

with open('result/' + dataset_name + '/FedAvg/OSMD/test_accu_list_nproviders=' + str(n_providers) + '_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(test_accu_list, f)
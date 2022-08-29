import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
from util import *

# the name of dataset
dataset_name = 'KMNIST'

print('dataset: {}'.format(dataset_name))

with open('data/' + dataset_name.lower() + '/corrupted_data.pickle', 'rb') as f:
    providers_train_list, val_loader, test_loader = pickle.load(f)

n_providers = len(providers_train_list)  # number of providers
step_size_local = 0.1  # step size for local updates
n_epoch = 1  # number of epochs for local computation of each round
batch_size = 10  # number of providers chosen in each communication round
n_commun = 1000  # number of communications

# intialize the number of access for each data provider
n_access = np.zeros(n_providers)

np.random.seed(111)  # set up the random seed

DSV_list = []  # List to store the DSV values
DSV = np.zeros(n_providers)  # initilaize the Data Shapley values
DSV_convg_measure = []  # check if DSV has converged
back_check = 10  # the gap to check convergence of DSV
c_tol = 0.01  # tolerance to decide if DSV has converged
max_n_perm = 500  # maximum number of permutations

start_time = time.time()
t  = 0
while True:
    # check if the number of permutations exceeds the maximum threshold
    t += 1
    if t > max_n_perm:
        break

    # shuffle all the training samples
    shuffled_seq = np.arange(n_providers)
    np.random.shuffle(shuffled_seq)

    # initialize the local model
    model_local = Net()
    optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
    
    v = 0.0  # intialize the value function
    for count_i, i in enumerate(shuffled_seq):
        # make local computation
        for e in range(n_epoch):
            for imgs, labels in providers_train_list[i]:
                optim_local.zero_grad()
                outs = model_local(imgs)
                loss = loss_CE_fn(outs, labels) / len(labels)
                loss.backward()
                optim_local.step()
        
        if count_i == 0:
            # intialize the value function
            v = -model_eval2(model_local, val_loader).item()
            continue

        # compute the local utility value
        v_new = -model_eval2(model_local, val_loader).item()

        # update the DSV
        DSV[i] = ((t-1) / t) * DSV[i] + (1 / t) * (v_new - v)

        # update the value function
        v = v_new

    # append the DSV to DSV list
    DSV_list.append(DSV.copy())

    # print out the DSV
    print(DSV)

    # compute the convergence criterion of DSV
    if t > back_check:
        s = 0.0
        count = 0
        for i in range(len(DSV)):
            if abs(DSV_list[-1][i]) > 0.0:
                count += 1
                s += abs(DSV_list[-1][i] - DSV_list[-1-back_check][i]) / abs(DSV_list[-1][i])
        s /= count
        DSV_convg_measure.append(s)

        if s < c_tol:
           break

    # print out useful information
    if len(DSV_convg_measure) > 0:
        print("The permutation round: {} | convergence criterion: {:.3f}".format(t, DSV_convg_measure[-1]))
    else:
        print("The permutation round: {}".format(t))

end_time = time.time()

# store the result
with open('result/' + dataset_name + '/FedAvg/DSV/time_cost.pickle', 'wb') as f:
    pickle.dump(end_time - start_time, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV_list.pickle', 'wb') as f:
    pickle.dump(DSV_list, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV_convg_measure.pickle', 'wb') as f:
    pickle.dump(DSV_convg_measure, f)

with open('result/' + dataset_name + '/FedAvg/DSV/DSV.pickle', 'wb') as f:
    pickle.dump(DSV, f)
 
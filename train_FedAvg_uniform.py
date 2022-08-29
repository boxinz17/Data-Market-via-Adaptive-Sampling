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
step_size_global = step_size_local  # step size for global updates
n_epoch = 1  # number of epochs for local computation of each round
batch_size = 10  # number of providers chosen in each communication round
n_commun = 500  # number of communications

# intialize the number of access for each data provider
n_access = np.zeros(n_providers)

np.random.seed(111)  # set up the random seed
model = Net()  # initialize the global model
util_list = []  # list storing validation lost
test_accu_list = []  # list storing test accuracy
start_time = time.time()
for t in range(n_commun):
    # shrink the optimization stepsize every 200 communication rounds
    if (t+1) % 200 == 0:
        step_size_local /= 2
        step_size_global = step_size_local
    
    # compute the performance of the global model on
    # hold-out test set
    test_accu = model_eval(model, test_loader)
    test_accu_list.append(test_accu)
    
    # compute the utility value of the global model
    util = model_eval2(model, val_loader)
    util_list.append(util.item())
    
    # print out useful information
    print('communication round: {} | evaluation loss: {} | test accuracy: {:.3f}%'.format(t+1, util, test_accu*100))
    print(n_access)
    
    # sample providers
    chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False)
    
    # store the current global model parameters
    global_params = []
    for f in model.parameters():
        global_params.append(f.data)
    
    updates = []  # store the local updates
    for i in chosen_providers:
        n_access[i] += 1
        local_updates = []
        
        # initialize the local model
        model_local = Net()
        model_local.load_state_dict(model.state_dict())
        optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
        
        # make local computation
        for e in range(n_epoch):
            for imgs, labels in providers_train_list[i]:
                optim_local.zero_grad()
                outs = model_local(imgs)
                loss = loss_CE_fn(outs, labels) / len(labels)
                loss.backward()
                optim_local.step()
        
        # compute the local updates
        for j, f in enumerate(model_local.parameters()):
            local_updates.append((f.data-global_params[j])/step_size_local)
        updates.append(local_updates)
    
    # make updates of the global model by FedAvg
    for j, f in enumerate(model.parameters()):
        update = updates[0][j]
        for i in range(1,batch_size):
            update += updates[i][j]
        update /= batch_size
        f.data.add_(step_size_global*update)

end_time = time.time()

# store the result
with open('result/' + dataset_name + '/FedAvg/uniform/model.pickle', 'wb') as f:
    pickle.dump(model, f)

with open('result/' + dataset_name + '/FedAvg/uniform/n_access.pickle', 'wb') as f:
    pickle.dump(n_access, f)

with open('result/' + dataset_name + '/FedAvg/uniform/util_list.pickle', 'wb') as f:
    pickle.dump(util_list, f)

with open('result/' + dataset_name + '/FedAvg/uniform/time_cost.pickle', 'wb') as f:
    pickle.dump(end_time - start_time, f)

with open('result/' + dataset_name + '/FedAvg/uniform/test_accu_list.pickle', 'wb') as f:
    pickle.dump(test_accu_list, f)
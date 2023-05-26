import numpy as np
import pickle
from util import *
import sys

# read the input random seed index
i_rand = int(sys.argv[1])

n_providers = 100

# load the random seed array
with open('rd_seed_array.pickle', 'rb') as f:
    rd_seed_array = pickle.load(f)

# load the data
with open('data/reg_param.pickle', 'rb') as f:
    reg_param = pickle.load(f)

with open('data/providers_train_list.pickle', 'rb') as f:
    providers_train_list = pickle.load(f)

with open('data/val_data.pickle', 'rb') as f:
    val_data = pickle.load(f)

step_size_global = 0.01  # step size for global updates
batch_size = int(0.1 * n_providers)  # number of providers chosen in each communication round
n_commun = 1500  # number of communications

w_true = reg_param[0, :]  # the true parameter that the consumer wants

# set up random seed
print("random seed: {}".format(rd_seed_array[i_rand]))
np.random.seed(rd_seed_array[i_rand])

w_est, n_access, util_list, est_err_list, time_used = train_FedAvg_uniform_mr(providers_train_list, val_data, w_true, step_size_global, batch_size, n_commun)

print("Time used: {}(s)".format(time_used))

# store the result
with open('result/uniform/w_est_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(w_est, f)

with open('result/uniform/n_access_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(n_access, f)

with open('result/uniform/util_list_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(util_list, f)

with open('result/uniform/est_err_list_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(est_err_list, f)

with open('result/uniform/time_used_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(time_used, f)
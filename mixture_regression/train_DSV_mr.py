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
back_check = 100  # the gap to check convergence of DSV
c_tol = 0.01  # tolerance to decide if DSV has converged
max_n_perm = 10000  # maximum number of permutations

w_true = reg_param[0, :]  # the true parameter that the consumer wants

# set up random seed
print("random seed: {}".format(rd_seed_array[i_rand]))
np.random.seed(rd_seed_array[i_rand])

DSV_list, DSV_convg_measure, DSV, time_used = train_DSV_mr(providers_train_list, val_data, step_size_global, back_check, c_tol, max_n_perm)

print("Time used: {}(s)".format(time_used))

# store the result
with open('result/DSV/DSV_list_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV_list, f)

with open('result/DSV/DSV_convg_measure_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV_convg_measure, f)

with open('result/DSV/DSV_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(DSV, f)

with open('result/DSV/time_used_rdseed=' + str(rd_seed_array[i_rand]) + '.pickle', 'wb') as f:
    pickle.dump(time_used, f)
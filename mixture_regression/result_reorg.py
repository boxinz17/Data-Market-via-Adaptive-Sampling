import numpy as np
import pickle

# the number of providers
n_providers = 100

# load the random seed array
with open('rd_seed_array.pickle', 'rb') as f:
    rd_seed_array = pickle.load(f)

n_rep = len(rd_seed_array)  # number of repetitions
n_commun = 1500  # number of communications

# reorganize the result from uniform
# initialize the result arrays
n_access_array = np.zeros((n_rep, n_providers))
est_err_array = np.zeros((n_rep, n_commun))

for i in range(n_rep):
    with open('result/uniform/n_access_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        n_access_array[i, :] = pickle.load(f)

    with open('result/uniform/est_err_list_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        est_err_array[i, :] = pickle.load(f)

# store the result
with open('result/uniform/n_access_array.pickle', 'wb') as f:
    pickle.dump(n_access_array, f)

with open('result/uniform/est_err_array.pickle', 'wb') as f:
    pickle.dump(est_err_array, f)

# reorganize the result from OSMD
# initialize the result arrays
n_access_array = np.zeros((n_rep, n_providers))
est_err_array = np.zeros((n_rep, n_commun))

for i in range(n_rep):
    with open('result/OSMD/n_access_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        n_access_array[i, :] = pickle.load(f)

for i in range(n_rep):
    with open('result/OSMD/est_err_list_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        est_err_array[i, :] = pickle.load(f)

# store the result
with open('result/OSMD/n_access_array.pickle', 'wb') as f:
    pickle.dump(n_access_array, f)

with open('result/OSMD/est_err_array.pickle', 'wb') as f:
    pickle.dump(est_err_array, f)

# reorganize the result from DSV
# initialize the result arrays
DSV_array = np.zeros((n_rep, n_providers))

for i in range(n_rep):
    with open('result/DSV/DSV_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        DSV_array[i, :] = pickle.load(f)

# store the result
with open('result/DSV/DSV_array.pickle', 'wb') as f:
    pickle.dump(DSV_array, f)


# reorganize the result from FedDSV
# initialize the result arrays
FedDSV_array = np.zeros((n_rep, n_providers))
est_err_array = np.zeros((n_rep, n_commun))

for i in range(n_rep):
    with open('result/FedDSV/FedDSV_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        FedDSV_array[i, :] = pickle.load(f)

for i in range(n_rep):
    with open('result/FedDSV/est_err_list_rdseed=' + str(rd_seed_array[i]) + '.pickle', 'rb') as f:
        est_err_array[i, :] = pickle.load(f)

# store the result
with open('result/FedDSV/FedDSV_array.pickle', 'wb') as f:
    pickle.dump(FedDSV_array, f)

with open('result/FedDSV/est_err_array.pickle', 'wb') as f:
    pickle.dump(est_err_array, f)
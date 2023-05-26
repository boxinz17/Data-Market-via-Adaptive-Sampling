import numpy as np
import pickle

np.random.seed(111)
rd_seed_array = np.random.randint(low=0, high=10000, size=10, dtype=int)

# store the result
with open('rd_seed_array.pickle', 'wb') as f:
    pickle.dump(rd_seed_array, f)
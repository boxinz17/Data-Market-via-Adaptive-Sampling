import numpy as np
import pickle

np.random.seed(123)
rd_seed_array =  np.random.choice(np.arange(10000), size=100, replace=False)

# store the result
with open('rd_seed_array.pickle', 'wb') as f:
    pickle.dump(rd_seed_array, f)
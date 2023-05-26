import numpy as np
import pickle

# set up problem parameters
n_providers = 100  # number of providers
n_train = 500  # number of training samples of each provider
n_val = 100  # number of validation samples

d = 1000  # dimension of regression parameters
n_group = 4  # number of mixture groups

# set up the random seed
rd_seed = 123

# generate the regression parameters
np.random.seed(rd_seed)
reg_param = np.zeros((n_group, d))
for i in range(n_group):
    reg_param[i, :] = np.random.uniform(low=i*0.5, high=(i+1)*0.5, size=d)

# generate the group label for each provider
# we use label 0 as the label of consumer
np.random.seed(rd_seed)
group_labels = np.random.choice(np.arange(n_group), size=n_providers)

# generate the training data
np.random.seed(rd_seed)
providers_train_list = []
for i in range(n_providers):
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(n_train, d))
    y_train = np.matmul(X_train, reg_param[group_labels[i], :]) + np.random.normal(loc=0.0, scale=0.5, size=n_train)
    providers_train_list.append((X_train.copy(), y_train.copy()))

# generate the validation data
X_val = np.random.normal(loc=0.0, scale=1.0, size=(n_val, d))
y_val = np.matmul(X_val, reg_param[0, :]) + np.random.normal(loc=0.0, scale=0.5, size=n_val)
val_data = (X_val, y_val)

# store the data
with open('data/reg_param.pickle', 'wb') as f:
    pickle.dump(reg_param, f)

with open('data/group_labels.pickle', 'wb') as f:
    pickle.dump(group_labels, f)

with open('data/providers_train_list.pickle', 'wb') as f:
    pickle.dump(providers_train_list, f)

with open('data/val_data.pickle', 'wb') as f:
    pickle.dump(val_data, f)
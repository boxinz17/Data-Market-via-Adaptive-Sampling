import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

# the list of names of datasets
dataset_name_list = [['MNIST', 'FMNIST'], ['KMNIST', 'CIFAR10']]

# the number of providers
n_providers = 100

# set up the plot path
plot_path = 'plots/'

# plot and save fig

std_plot = 1.0  # the scale parameter of standard error

# load the result
n_access_array_uniform_mean_list = []
n_access_array_uniform_std_list = []
util_array_uniform_mean_list = []
util_array_uniform_std_list = []
test_accu_array_uniform_mean_list = []
test_accu_array_uniform_std_list = []
n_access_array_OSMD_mean_list = []
n_access_array_OSMD_std_list = []
util_array_OSMD_mean_list = []
util_array_OSMD_std_list = []
test_accu_array_OSMD_mean_list = []
test_accu_array_OSMD_std_list = []
DSV_array_mean_list = []
DSV_array_std_list = []
FedDSV_array_mean_list = []
FedDSV_array_std_list = []
test_accu_array_FedDSV_mean_list = []
test_accu_array_FedDSV_std_list = []

for i in range(2):
    for j in range(2):
        # the name of dataset
        dataset_name = dataset_name_list[i][j]

        # set up the result path
        result_OSMD_path = 'result/' + dataset_name + '/FedAvg/OSMD'
        result_uniform_path = 'result/' + dataset_name + '/FedAvg/uniform'
        result_DSV_path = 'result/' + dataset_name + '/FedAvg/DSV'
        result_FedDSV_path = 'result/' + dataset_name + '/FedAvg/FedDSV'

        # load the result
        # load uniform result
        with open(result_uniform_path + '/n_access_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            n_access_array_uniform = pickle.load(f)

        with open(result_uniform_path + '/util_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            util_array_uniform = pickle.load(f)

        with open(result_uniform_path + '/test_accu_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            test_accu_array_uniform = pickle.load(f)

        # load OSMD result
        with open(result_OSMD_path + '/n_access_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            n_access_array_OSMD = pickle.load(f)

        with open(result_OSMD_path + '/util_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            util_array_OSMD = pickle.load(f)

        with open(result_OSMD_path + '/test_accu_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            test_accu_array_OSMD = pickle.load(f)

        # load DSV result
        with open(result_DSV_path + '/DSV_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            DSV_array = pickle.load(f)

        # load FedDSV result
        with open(result_FedDSV_path + '/FedDSV_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            FedDSV_array = pickle.load(f)
        
        with open(result_FedDSV_path + '/test_accu_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            test_accu_array_FedDSV = pickle.load(f)

        # compute the mean and standard error
        n_access_array_uniform_mean = n_access_array_uniform.mean(0)
        n_access_array_uniform_std = n_access_array_uniform.std(0)

        n_access_array_uniform_mean_list.append(n_access_array_uniform_mean)
        n_access_array_uniform_std_list.append(n_access_array_uniform_std)

        util_array_uniform_mean = util_array_uniform.mean(0)
        util_array_uniform_std = util_array_uniform.std(0)

        util_array_uniform_mean_list.append(util_array_uniform_mean)
        util_array_uniform_std_list.append(util_array_uniform_std)

        test_accu_array_uniform_mean = test_accu_array_uniform.mean(0)
        test_accu_array_uniform_std = test_accu_array_uniform.std(0)

        test_accu_array_uniform_mean_list.append(test_accu_array_uniform_mean)
        test_accu_array_uniform_std_list.append(test_accu_array_uniform_std)

        n_access_array_OSMD_mean = n_access_array_OSMD.mean(0)
        n_access_array_OSMD_std = n_access_array_OSMD.std(0)

        n_access_array_OSMD_mean_list.append(n_access_array_OSMD_mean)
        n_access_array_OSMD_std_list.append(n_access_array_OSMD_std)

        util_array_OSMD_mean = util_array_OSMD.mean(0)
        util_array_OSMD_std = util_array_OSMD.std(0)

        util_array_OSMD_mean_list.append(util_array_OSMD_mean)
        util_array_OSMD_std_list.append(util_array_OSMD_std)

        test_accu_array_OSMD_mean = test_accu_array_OSMD.mean(0)
        test_accu_array_OSMD_std = test_accu_array_OSMD.std(0)

        test_accu_array_OSMD_mean_list.append(test_accu_array_OSMD_mean)
        test_accu_array_OSMD_std_list.append(test_accu_array_OSMD_std)

        DSV_array_mean = DSV_array.mean(0)
        DSV_array_std = DSV_array.std(0)

        DSV_array_mean_list.append(DSV_array_mean)
        DSV_array_std_list.append(DSV_array_std)

        FedDSV_array_mean = FedDSV_array.mean(0)
        FedDSV_array_std = FedDSV_array.std(0)

        FedDSV_array_mean_list.append(FedDSV_array_mean)
        FedDSV_array_std_list.append(FedDSV_array_std)

        test_accu_array_FedDSV_mean = test_accu_array_FedDSV.mean(0)
        test_accu_array_FedDSV_std = test_accu_array_FedDSV.std(0)

        test_accu_array_FedDSV_mean_list.append(test_accu_array_FedDSV_mean)
        test_accu_array_FedDSV_std_list.append(test_accu_array_FedDSV_std)

# plot monetary allocation (revenue allocation)
batch_size = int(0.1 * n_providers)  # number of providers chosen in each communication round
fig, axes = plt.subplots(2, 2, figsize=[36., 24.])
for i in range(2):
    for j in range(2):
        dataset_name = dataset_name_list[i][j]

        axes[i, j].set_title(dataset_name, fontsize=60)

        if dataset_name == 'CIFAR10':
            n_commun = 1000  # number of communications
        else:
            n_commun = 500  # number of communications

        total_budget = n_commun * batch_size

        # set up the result path
        result_uniform_path = 'result/' + dataset_name + '/FedAvg/uniform'
        result_OSMD_path = 'result/' + dataset_name + '/FedAvg/OSMD'
        result_DSV_path = 'result/' + dataset_name + '/FedAvg/DSV'
        result_FedDSV_path = 'result/' + dataset_name + '/FedAvg/FedDSV'

        # load uniform result
        with open(result_uniform_path + '/n_access_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            n_access_array_uniform = pickle.load(f)

        # load OSMD result
        with open(result_OSMD_path + '/n_access_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            n_access_array_OSMD = pickle.load(f)

        # load DSV result
        with open(result_DSV_path + '/DSV_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            DSV_array = pickle.load(f)

        # load FedDSV result
        with open(result_FedDSV_path + '/FedDSV_array_nproviders=' + str(n_providers) + '.pickle', 'rb') as f:
            FedDSV_array = pickle.load(f)

        money_uniform_array = np.zeros(n_access_array_uniform.shape)
        for k in range(money_uniform_array.shape[0]):
            money_uniform_array[k, :] = (n_access_array_uniform[k, :] / n_access_array_uniform[k, :].sum()) * total_budget
        money_uniform_array_mean = money_uniform_array.mean(0)
        money_uniform_array_std = money_uniform_array.std(0)

        money_OSMD_array = np.zeros(n_access_array_OSMD.shape)
        for k in range(money_OSMD_array.shape[0]):
            money_OSMD_array[k, :] = (n_access_array_OSMD[k, :] / n_access_array_OSMD[k, :].sum()) * total_budget
        money_OSMD_array_mean = money_OSMD_array.mean(0)
        money_OSMD_array_std = money_OSMD_array.std(0)

        money_DSV_array = np.zeros(DSV_array.shape)
        for k in range(money_DSV_array.shape[0]):
            DSV_shift = DSV_array[k, :] - DSV_array[k, :].min()
            money_DSV_array[k, :] = (DSV_shift /  DSV_shift.sum()) * total_budget
        money_DSV_array_mean = money_DSV_array.mean(0)
        money_DSV_array_std = money_DSV_array.std(0)
        
        money_FedDSV_array = np.zeros(FedDSV_array.shape)
        for k in range(money_FedDSV_array.shape[0]):
            FedDSV_shift = FedDSV_array[k, :] - FedDSV_array[k, :].min()
            money_FedDSV_array[k, :] = (FedDSV_shift /  FedDSV_shift.sum()) * total_budget
        money_FedDSV_array_mean = money_FedDSV_array.mean(0)
        money_FedDSV_array_std = money_FedDSV_array.std(0)

        handle_uniform = axes[i, j].errorbar(np.arange(n_providers), money_uniform_array_mean, yerr=std_plot*money_uniform_array_std, color="r", linestyle='--', marker='o', linewidth=3, label='uniform', capsize=5)
        handle_OSMD = axes[i, j].errorbar(np.arange(n_providers), money_OSMD_array_mean, yerr=std_plot*money_OSMD_array_std, color="b", linestyle='--', marker='o', linewidth=3, label='OSMD', capsize=5)
        handle_DSV = axes[i, j].errorbar(np.arange(n_providers), money_DSV_array_mean, yerr=std_plot*money_DSV_array_std, color="c", linestyle='--', marker='o', linewidth=3, label='G-Shapley', capsize=5)
        handle_FedDSV = axes[i, j].errorbar(np.arange(n_providers), money_FedDSV_array_mean, yerr=std_plot*money_FedDSV_array_std, color="m", linestyle='--', marker='o', linewidth=3, label='FedSV-PS', capsize=5)

        if i == 1:
            axes[i, j].set_xlabel("provider id", fontsize=60)
        if j == 0:
            axes[i, j].set_ylabel("money", fontsize=60)

        axes[i, j].set_xticks([1]+list(np.arange(10, n_providers+10, 10)))
        axes[i, j].tick_params(axis='x', labelsize=40)
        axes[i, j].tick_params(axis='y', labelsize=40)

fig.legend(handles=[handle_OSMD, handle_uniform, handle_DSV, handle_FedDSV], loc='upper center', ncol=4, fontsize=55, frameon=False)
fig.savefig(plot_path+'money.png', dpi=100, bbox_inches='tight')
fig.savefig(plot_path+'money.pdf', dpi=100, bbox_inches='tight')
plt.close(fig)

# plot the hold-out accuracy (budget allocation)
fig, axes = plt.subplots(2, 2, figsize=[36., 24.])
for i in range(2):
    for j in range(2):
        dataset_name = dataset_name_list[i][j]

        n_commun = len(test_accu_array_OSMD_mean_list[i*2+j])

        handle_OSMD, = axes[i, j].plot(np.arange(n_commun), test_accu_array_OSMD_mean_list[i*2+j], color="b", linestyle='-', linewidth=3, label='OSMD')
        axes[i, j].fill_between(np.arange(n_commun), test_accu_array_OSMD_mean_list[i*2+j]-std_plot*test_accu_array_OSMD_std_list[i*2+j], test_accu_array_OSMD_mean_list[i*2+j]+std_plot*test_accu_array_OSMD_std_list[i*2+j], facecolor='b', edgecolor='b', alpha=0.2, linestyle='dashdot')
        
        handle_uniform, = axes[i, j].plot(np.arange(n_commun), test_accu_array_uniform_mean_list[i*2+j], color="r", linestyle='-', linewidth=3, label='uniform')
        axes[i, j].fill_between(np.arange(n_commun), test_accu_array_uniform_mean_list[i*2+j]-std_plot*test_accu_array_uniform_std_list[i*2+j], test_accu_array_uniform_mean_list[i*2+j]+std_plot*test_accu_array_uniform_std_list[i*2+j], facecolor='r', edgecolor='r', alpha=0.2, linestyle='dashdot')
        
        axes[i, j].set_title(dataset_name, fontsize=60)
        
        if i == 1:
            axes[i, j].set_xlabel("# Communication Round", fontsize=60)
        if j == 0:
            axes[i, j].set_ylabel("Test Accuracy", fontsize=60)

        axes[i, j].set_xticks([1]+list(np.arange(100, n_commun+100, 100)))
        axes[i, j].tick_params(axis='x', labelsize=35)
        axes[i, j].tick_params(axis='y', labelsize=40)

fig.legend(handles=[handle_OSMD, handle_uniform], loc='upper center', ncol=3, fontsize=55, frameon=False)
fig.savefig(plot_path+'test_accu.png', dpi=100, bbox_inches='tight')
fig.savefig(plot_path+'test_accu.pdf', dpi=100, bbox_inches='tight')
plt.close(fig)
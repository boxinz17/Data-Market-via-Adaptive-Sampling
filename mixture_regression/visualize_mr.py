import matplotlib.pyplot as plt
import numpy as np
import pickle

# set up parameters
n_providers = 100
batch_size = int(0.1 * n_providers)  # number of providers chosen in each communication round
n_commun = 1500  # number of communications
total_budget = n_commun * batch_size

# set up the result path
result_uniform_path = 'result/uniform'
result_OSMD_path = 'result/OSMD'
result_DSV_path = 'result/DSV'
result_FedDSV_path = 'result/FedDSV'
plot_path = 'plot/'

# load group labels
with open('data/group_labels.pickle', 'rb') as f:
    group_labels = pickle.load(f)

# load uniform result
with open(result_uniform_path + '/n_access_array.pickle', 'rb') as f:
    n_access_array_uniform = pickle.load(f)

with open(result_uniform_path + '/est_err_array.pickle', 'rb') as f:
    est_err_array_uniform = pickle.load(f)

# load OSMD result
with open(result_OSMD_path + '/n_access_array.pickle', 'rb') as f:
    n_access_array_OSMD = pickle.load(f)

with open(result_OSMD_path + '/est_err_array.pickle', 'rb') as f:
    est_err_array_OSMD = pickle.load(f)

# load DSV result
with open(result_DSV_path + '/DSV_array.pickle', 'rb') as f:
    DSV_array = pickle.load(f)

# load FedDSV result
with open(result_FedDSV_path + '/FedDSV_array.pickle', 'rb') as f:
    FedDSV_array = pickle.load(f)

with open(result_FedDSV_path + '/est_err_array.pickle', 'rb') as f:
    est_err_array_FedDSV = pickle.load(f)

# plot
fig, axes = plt.subplots(2, 1, figsize=[36., 24.])
std_plot = 1.0

# plot monetary allocation
ind_sort = np.argsort(group_labels)
group_labels[ind_sort]

money_uniform_array = np.zeros(n_access_array_uniform.shape)
for k in range(money_uniform_array.shape[0]):
    money_uniform_array[k, :] = (n_access_array_uniform[k, :] / n_access_array_uniform[k, :].sum()) * total_budget
money_uniform_array_mean = money_uniform_array.mean(0)
money_uniform_array_std = money_uniform_array.std(0)
money_uniform_array_25p = np.percentile(money_uniform_array, q=0.25, axis=0)
money_uniform_array_75p = np.percentile(money_uniform_array, q=0.75, axis=0)

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
money_DSV_array_25p = np.percentile(money_DSV_array, q=0.25, axis=0)
money_DSV_array_75p = np.percentile(money_DSV_array, q=0.75, axis=0)

money_FedDSV_array = np.zeros(FedDSV_array.shape)
for k in range(money_FedDSV_array.shape[0]):
    FedDSV_shift = FedDSV_array[k, :] - FedDSV_array[k, :].min()
    money_FedDSV_array[k, :] = (FedDSV_shift /  FedDSV_shift.sum()) * total_budget
money_FedDSV_array_mean = money_FedDSV_array.mean(0)
money_FedDSV_array_std = money_FedDSV_array.std(0)
money_FedDSV_array_25p = np.percentile(money_FedDSV_array, q=0.25, axis=0)
money_FedDSV_array_75p = np.percentile(money_FedDSV_array, q=0.75, axis=0)

handle_uniform = axes[0].errorbar(np.arange(n_providers), money_uniform_array_mean[ind_sort], yerr=money_uniform_array_std[ind_sort], color="r", linestyle='--', marker='o', linewidth=3, label='uniform', capsize=3)
handle_OSMD = axes[0].errorbar(np.arange(n_providers), money_OSMD_array_mean[ind_sort], yerr=std_plot*money_OSMD_array_std[ind_sort], color="b", linestyle='--', marker='o', linewidth=3, label='OSMD', capsize=3)
handle_DSV = axes[0].errorbar(np.arange(n_providers), money_DSV_array_mean[ind_sort], yerr=std_plot*money_DSV_array_std[ind_sort], color="c", linestyle='--', marker='o', linewidth=3, label='G-Shapley', capsize=3)
handle_FedDSV = axes[0].errorbar(np.arange(n_providers), money_FedDSV_array_mean[ind_sort], yerr=std_plot*money_FedDSV_array_std[ind_sort], color="m", linestyle='--', marker='o', linewidth=3, label='FedSV-PS', capsize=3)

axes[0].set_xlabel("provider's group", fontsize=60)
axes[0].set_ylabel("money", fontsize=60)

axes[0].set_xticks(np.arange(n_providers))
axes[0].set_xticklabels(group_labels[ind_sort]+1)
axes[0].tick_params(axis='x', labelsize=30)
axes[0].tick_params(axis='y', labelsize=35)

axes[0].legend(handles=[handle_OSMD, handle_uniform, handle_DSV, handle_FedDSV], loc='upper right', ncol=4, fontsize=55, frameon=False)

# plot estimation error
est_err_array_uniform_mean = est_err_array_uniform.mean(0)
est_err_array_uniform_std = est_err_array_uniform.std(0)

est_err_array_OSMD_mean = est_err_array_OSMD.mean(0)
est_err_array_OSMD_std = est_err_array_OSMD.std(0)

est_err_array_FedDSV_mean = est_err_array_FedDSV.mean(0)
est_err_array_FedDSV_std = est_err_array_FedDSV.std(0)

handle_uniform, = axes[1].plot(np.arange(n_commun), est_err_array_uniform_mean, color="r", linestyle='-', linewidth=3, label='uniform')
axes[1].fill_between(np.arange(n_commun), est_err_array_uniform_mean-std_plot*est_err_array_uniform_std, est_err_array_uniform_mean+std_plot*est_err_array_uniform_std, facecolor='r', edgecolor='r', alpha=0.2, linestyle='dashdot')

handle_OSMD, = axes[1].plot(np.arange(n_commun), est_err_array_OSMD_mean, color="b", linestyle='-', linewidth=3, label='OSMD')
axes[1].fill_between(np.arange(n_commun), est_err_array_OSMD_mean-std_plot*est_err_array_OSMD_std, est_err_array_OSMD_mean+std_plot*est_err_array_OSMD_std, facecolor='b', edgecolor='b', alpha=0.2, linestyle='dashdot')

axes[1].set_xlabel("# communication round", fontsize=60)
axes[1].set_ylabel("estimation error", fontsize=60)

axes[1].tick_params(axis='x', labelsize=35)
axes[1].tick_params(axis='y', labelsize=35)

axes[1].legend(handles=[handle_OSMD, handle_uniform], loc='upper center', ncol=3, fontsize=55, frameon=False)

fig.savefig(plot_path+'mixture_regression.png', dpi=100, bbox_inches='tight')
fig.savefig(plot_path+'mixture_regression.pdf', dpi=100, bbox_inches='tight')
plt.close(fig)
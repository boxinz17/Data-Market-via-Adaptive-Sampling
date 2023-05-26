import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

# set up the plot path
plot_path = '../plots/'

# the list of names of datasets
dataset_name_list = [['MNIST', 'FMNIST'], ['KMNIST', 'CIFAR10']]

fig, axes = plt.subplots(2, 2, figsize=[36., 24.])
for i in range(2):
    for j in range(2):
        dataset_name = dataset_name_list[i][j]

        # set up the result path
        result_OSMD_path = 'result/' + dataset_name + '/FedAvg/OSMD'
        result_uniform_path = 'result/' + dataset_name + '/FedAvg/uniform'
        result_DSV_path = 'result/' + dataset_name + '/FedAvg/DSV'
        result_FedDSV_path = 'result/' + dataset_name + '/FedAvg/FedDSV'

        # load the result
        with open(result_uniform_path + '/time_used_array.pickle', 'rb') as f:
            time_used_array_uniform = pickle.load(f)

        with open(result_OSMD_path + '/time_used_array.pickle', 'rb') as f:
            time_used_array_OSMD = pickle.load(f)

        with open(result_DSV_path + '/time_used_array.pickle', 'rb') as f:
            time_used_array_DSV = pickle.load(f)

        with open(result_FedDSV_path + '/time_used_array.pickle', 'rb') as f:
            time_used_array_FedDSV = pickle.load(f)

        # compute the mean and standard error
        time_used_array_uniform_mean = time_used_array_uniform.mean(0)
        time_used_array_uniform_std = time_used_array_uniform.std(0)

        time_used_array_OSMD_mean = time_used_array_OSMD.mean(0)
        time_used_array_OSMD_std = time_used_array_OSMD.std(0)

        time_used_array_DSV_mean = time_used_array_DSV.mean(0)
        time_used_array_DSV_std = time_used_array_DSV.std(0)

        time_used_array_FedDSV_mean = time_used_array_FedDSV.mean(0)
        time_used_array_FedDSV_std = time_used_array_FedDSV.std(0)

        # plot
        std_plot = 1.0  # the scale parameter of standard error

        handle_OSMD = axes[i, j].errorbar([50, 100, 200, 400], time_used_array_OSMD_mean, yerr=std_plot*time_used_array_OSMD_std, color="b", linestyle='--', marker='o', linewidth=3, label='OSMD', capsize=5)
        handle_uniform = axes[i, j].errorbar([50, 100, 200, 400], time_used_array_uniform_mean, yerr=std_plot*time_used_array_uniform_std, color="r", linestyle='--', marker='o', linewidth=3, label='uniform', capsize=5)
        handle_DSV = axes[i, j].errorbar([50, 100, 200, 400], time_used_array_DSV_mean, yerr=std_plot*time_used_array_DSV_std, color="c", linestyle='--', marker='o', linewidth=3, label='G-Shapley', capsize=5)
        handle_FedDSV = axes[i, j].errorbar([50, 100, 200, 400], time_used_array_FedDSV_mean, yerr=std_plot*time_used_array_FedDSV_std, color="m", linestyle='--', marker='o', linewidth=3, label='FedSV-PS', capsize=5)

        axes[i, j].set_title(dataset_name, fontsize=60)
        
        if i == 1:
            axes[i, j].set_xlabel("# providers", fontsize=60)
        if j == 0:
            axes[i, j].set_ylabel("Wallclock Time (s)", fontsize=60)

        axes[i, j].tick_params(axis='x', labelsize=45)
        axes[i, j].tick_params(axis='y', labelsize=45)

fig.legend(handles=[handle_OSMD, handle_uniform, handle_DSV, handle_FedDSV], loc='upper center', ncol=4, fontsize=55, frameon=False)
fig.savefig(plot_path+'time_used.png', dpi=100, bbox_inches='tight')
fig.savefig(plot_path+'time_used.pdf', dpi=100, bbox_inches='tight')
plt.close(fig)
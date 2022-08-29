import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

# the name of task
dataset_name = 'CIFAR10'

# set up the result and plot path
result_OSMD_path = 'result/' + dataset_name + '/FedAvg/OSMD'
result_uniform_path = 'result/' + dataset_name + '/FedAvg/uniform'
result_DSV_path = 'result/' + dataset_name + '/FedAvg/DSV'
result_FedDSV_path = 'result/' + dataset_name + '/FedAvg/FedDSV'
plot_path = 'plots/' + dataset_name + '/FedAvg/'

# plot the access and evaluation result

# load the result
with open(result_OSMD_path + '/n_access.pickle', 'rb') as f:
    n_access_OSMD = pickle.load(f)

with open(result_uniform_path + '/n_access.pickle', 'rb') as f:
    n_access_uniform = pickle.load(f)

with open(result_DSV_path + '/DSV.pickle', 'rb') as f:
    DSV = pickle.load(f)

with open(result_FedDSV_path + '/FedDSV.pickle', 'rb') as f:
    FedDSV = pickle.load(f)

# plot and save fig
fig, axes = plt.subplots(1, 1, figsize=[24., 12.])
axes.plot(np.arange(len(n_access_OSMD)), n_access_OSMD, color="b", linestyle='--', marker='o', linewidth=3, label='OSMD')
axes.plot(np.arange(len(n_access_uniform)), n_access_uniform, color="r", linestyle='--', marker='o', linewidth=3, label='uniform')
axes.set_title(dataset_name, fontsize=25)

axes.set_xlabel("provider id", fontsize=25)
axes.set_ylabel("# access", fontsize=25)

axes.set_xticks([1]+list(np.arange(10, len(n_access_OSMD)+10, 10)))
axes.tick_params(axis='x', labelsize=25)
axes.tick_params(axis='y', labelsize=25)

ax2 = axes.twinx()
ax2.plot(DSV, color="c", linestyle='--', marker='o', linewidth=3, label='DSV')

# rescale FedDSV
FedDSV_rescale = (FedDSV - FedDSV.min()) / (FedDSV.max() - FedDSV.min())
FedDSV_rescale = FedDSV_rescale * (DSV.max()-DSV.min()) + DSV.min()

ax2.plot(FedDSV_rescale, color="m", linestyle='--', marker='o', linewidth=3, label='FedDSV')
ax2.set_ylabel("Data Shapley value", fontsize=25)
ax2.tick_params(axis='y', labelsize=25)

fig.legend(labels=['OSMD', 'uniform', 'DSV', 'FedDSV'], loc='upper right', ncol=4, fontsize=25)
fig.savefig(plot_path+'n_access.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# plot the hold-out accuracy

# load the result
with open(result_OSMD_path + '/test_accu_list.pickle', 'rb') as f:
    test_accu_list_OSMD = pickle.load(f)

with open(result_uniform_path + '/test_accu_list.pickle', 'rb') as f:
    test_accu_list_uniform = pickle.load(f)

# plot and save fig
fig, axes = plt.subplots(1, 1, figsize=[24., 12.])
axes.plot(np.arange(1, len(test_accu_list_OSMD)+1), test_accu_list_OSMD, color="b", linestyle='-', linewidth=3, label='OSMD')
axes.plot(np.arange(1, len(test_accu_list_uniform)+1), test_accu_list_uniform, color="r", linestyle='-', linewidth=3, label='uniform')
axes.set_title(dataset_name, fontsize=25)

axes.set_xlabel("# Communication Round", fontsize=25)
axes.set_ylabel("Test Accuracy", fontsize=25)

axes.set_xticks([1]+list(np.arange(100, len(test_accu_list_OSMD)+100, 100)))
axes.tick_params(axis='x', labelsize=25)
axes.tick_params(axis='y', labelsize=25)

fig.legend(labels=['OSMD', 'uniform'], loc='upper right', ncol=1, fontsize=25)
fig.savefig(plot_path+'test_accu.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# plot OSMD, uniform and FedDSV v.s. DSV
fig, axes = plt.subplots(1, 1, figsize=[24., 12.])
ind_sort = np.argsort(DSV)
axes.plot(DSV[ind_sort], n_access_OSMD[ind_sort], markerfacecolor='none', markeredgecolor="b", linestyle='None', marker='s', ms=15, label='OSMD')
axes.plot(DSV[ind_sort], n_access_uniform[ind_sort], markerfacecolor='none', markeredgecolor="r", linestyle='None', marker='o', ms=15, label='uniform')
axes.set_title(dataset_name, fontsize=25)

axes.set_xlabel("DSV", fontsize=25)
axes.set_ylabel("# access/FedDSV", fontsize=25)

axes.tick_params(axis='x', labelsize=25)
axes.tick_params(axis='y', labelsize=25)

ax2 = axes.twinx()
ax2.plot(DSV[ind_sort], FedDSV[ind_sort], markerfacecolor='none', markeredgecolor="m", linestyle='None', marker='v', ms=15, label='FedDSV')
ax2.set_ylabel("Federated Data Shapley value", fontsize=25)
ax2.tick_params(axis='y', labelsize=25)

fig.legend(labels=['OSMD', 'uniform', 'FedDSV'], loc='upper right', ncol=1, fontsize=25)
fig.savefig(plot_path+'OSMD_DSV.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# write the model information

# load the result
with open(result_OSMD_path + '/time_cost.pickle', 'rb') as f:
    time_cost_OSMD = pickle.load(f)

with open(result_uniform_path + '/time_cost.pickle', 'rb') as f:
    time_cost_uniform = pickle.load(f)

with open(result_DSV_path + '/time_cost.pickle', 'rb') as f:
    time_cost_DSV = pickle.load(f)

with open(result_FedDSV_path + '/time_cost.pickle', 'rb') as f:
    time_cost_FedDSV = pickle.load(f)

corr_OSMD, p_value_OSMD = scipy.stats.spearmanr(n_access_OSMD, DSV)
corr_uniform, p_value_uniform = scipy.stats.spearmanr(n_access_uniform, DSV)
corr_FedDSV, p_value_FedDSV = scipy.stats.spearmanr(FedDSV, DSV)

with open(plot_path + '/model_info.txt', 'w') as f:
    f.write("final testing accuracy of OSMD: {}".format(test_accu_list_OSMD[-1]))
    f.write('\n')
    f.write("final testing accuracy of uniform: {}".format(test_accu_list_uniform[-1]))
    f.write('\n\n')
    f.write('Total Wallclock Time of OSMD: {:.2f}(s)'.format(time_cost_OSMD))
    f.write('\n')
    f.write('Total Wallclock Time of Uniform: {:.2f}(s)'.format(time_cost_uniform))
    f.write('\n')
    f.write('Total Wallclock Time of DSV: {:.2f}(s)'.format(time_cost_DSV))
    f.write('\n')
    f.write('Total Wallclock Time of FedDSV: {:.2f}(s)'.format(time_cost_FedDSV))
    f.write('\n\n')
    f.write('The Spearman correlation between DSV and OSMD: {:.3f} | p-value: {}'.format(corr_OSMD, p_value_OSMD))
    f.write('\n')
    f.write('The Spearman correlation between DSV and uniform: {:.3f} | p-value: {}'.format(corr_uniform, p_value_uniform))
    f.write('\n')
    f.write('The Spearman correlation between DSV and FedDSV: {:.3f} | p-value: {}'.format(corr_FedDSV, p_value_FedDSV))
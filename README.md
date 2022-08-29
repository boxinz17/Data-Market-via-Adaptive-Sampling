# FL-Data-Market

Instruction to reproduce the experimental results:
1. Run data_load.py. The code will automatically download the dataset for you.
2. Run train_FedAvg_DSV.py, train_FedAvg_FedDSV.py, train_FedAvg_OSMD.py, train_FedAvg_uniform.py separately. Note that the user needs to manually set dataset_name variable, the acceptable value needs to be one of ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10'].
3. Run visualize_FedAvg.py to get the plots. Again, the user needs to manually set dataset_name variable, the acceptable value needs to be one of ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10'].
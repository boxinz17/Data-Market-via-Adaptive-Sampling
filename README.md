# Data-Market-via-Adaptive-Sampling

## Introduction

This repo aims to provide the code and instructions to reproduce the experimental results in the following paper 

> *Boxin Zhao, Boxiang Lyu, Raul Castro Fernandez. Mladen Kolar. Addressing Budget Allocation and Revenue Allocation in Data Market Environment Using an Adaptive Sampling Algorithm. International Conference on Machine Learning, 2023.* 

Please refer the paper for more details.

## Preparation

1. Run data_load.py. The code will automatically download the dataset for you.
> python data_load.py
2. Run rand_seed_generator.py to generate the list of random seeds.
> python rand_seed_generator.py

## Budget Allocation and Revenue Allocation

1. Run train_DSV.py, train_FedAvg_FedDSV.py, train_FedAvg_OSMD.py, train_FedAvg_uniform.py individually. Note that the user needs to manually set dataset_name variable, the acceptable value needs to be one of ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10']. Besides, user also needs to set the index of the list of random seeds, the acceptable variable is from 0-9.
> python train_FedAvg_uniform.py 0
2. After running all methods with all four datasets and all 10 random seeds, run result_reorg.py to reorganize the results, and then run visualize_FedAvg.py to get the plots of budget allocation and revenue allocation.
> python result_reorg.py \
> python visualize_revenue_budget.py

## Time Analysis
1. Change to time_analysis directory
2. Run data_load_time.py
> python data_load_time.py
3. Run train_DSV_time.py, train_FedAvg_FedDSV_time.py, train_FedAvg_OSMD_time.py, train_FedAvg_uniform_time.py individually. Note that the user needs to manually set dataset_name variable, the acceptable value needs to be one of ['MNIST', 'KMNIST', 'FMNIST', 'CIFAR10'].
Besides, user also needs to set the number of providers by setting the variable n_providers. In the paper, we choose n_providers=50, 100, 200, 400.
Finally, user also needs to set the index of the list of random seeds, the acceptable variable is from 0-9.
> python train_FedAvg_uniform_time.py 5
4. After running all methods with all four datasets, all data provider numbers (50, 100, 200 and 400) and all 10 random seeds, run result_reorg.py to reorganize the results, and then run visualize_time.py get the plots of time analysis.
> python result_reorg.py \
> python visualize_time.py

## Mixture Linear Regression
1. Change to mixture_regression directory
2. Run rand_seed_generator.py to generate the list of random seeds.
> python rand_seed_generator.py
3. Run data_generate.py to generate the data
> python data_generate.py
4. Run train_DSV_mr.py, train_FedAvg_FedDSV_mr.py, train_FedAvg_OSMD_mr.py, and train_FedAvg_uniform_mr.py individually. User needs to set the index of the list of random seeds, the acceptable variable is from 0-99.
> python train_FedAvg_uniform_mr.py 87
5. After running all methods with all random seeds, run result_reorg.py to reorganize the results, and then run visualize_mr.py get the plots of time analysis.
> python result_reorg.py \
> python visualize_mr.py

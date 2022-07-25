import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
import pickle

def data_prepare(dataset_name, n_devices, n_train, n_val, n_test, batch_size=64, rd_seed=111):
    """
    Return the train_loader_list, devices_train_list, val_loader_list and devices_val_list
    based on the dataset_name.
    """
    # Input: 
    #   dataset_name: A string, should be one of
    #   {"MNIST", "KMNIST", "FMNIST"}
    #   n_devices: Integer, number of devices
    #   n_train_list: List of integers, number of training samples of each device
    #   n_val: Inetger, number of validation samples used to compute utility
    #   n_test: Integer, number of held-out testing samples
    #   rd_seed: Integer, random seed
    # Return:
    #   (devices_train_list, val_loader, test_loader)
    np.random.seed(rd_seed)

    if dataset_name == "MNIST":
        data_path = 'data/mnist/'
        transform_data = datasets.MNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
                #transforms.Normalize((0.1307),(0.3081))
            ]))
    elif dataset_name == "KMNIST":
        data_path = 'data/kmnist/'
        transform_data = datasets.KMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1918),(0.3483))
            ]))
    elif dataset_name == "FMNIST":
        data_path = 'data/fmnist/'
        transform_data = datasets.FashionMNIST(
            data_path, train=True, download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.2861),(0.3530))
            ]))

    devices_train_list = []  # list of training data for devices
    val_loader = []
    test_loader = []

    sample_order = np.arange(len(transform_data))
    np.random.shuffle(sample_order)

    count = 0
    # Choose training data
    for i in range(n_devices):
        device_data = []
        for j in range(n_train):
            img, label = transform_data[sample_order[count]]
            # corrupt the data
            if j < (i // 10) * 0.2 * n_train:
                label_true = label  # note that label is int, thus no need to copy
                while label == label_true:
                    label = np.random.choice(10)
            device_data.append((img / torch.norm(img.squeeze(0)).item(), label))
            count += 1
        device_data = torch.utils.data.DataLoader(device_data, batch_size=batch_size, shuffle=True)
        devices_train_list.append(device_data)
    
    # Choose validation data
    val_loader = []
    for j in range(n_val):
        img, label = transform_data[sample_order[count]]
        val_loader.append((img / torch.norm(img.squeeze(0)).item(), label))
        count += 1
    val_loader = torch.utils.data.DataLoader(val_loader, batch_size=batch_size, shuffle=True)

    # Choose test data
    test_loader = []
    for j in range(n_test):
        img, label = transform_data[sample_order[count]]
        test_loader.append((img / torch.norm(img.squeeze(0)).item(), label))
        count += 1
    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=True)

    print(count)

    return devices_train_list, val_loader, test_loader

if __name__ == '__main__':
    dataset_name = 'MNIST'
    n_devices = 50
    n_train = 1000
    n_val = 1000
    n_test = 9000
    devices_train_list, val_loader, test_loader = data_prepare(dataset_name, n_devices, n_train, n_val, n_test, batch_size=5, rd_seed=111)
    with open('data/mnist/corrupted_data.pickle', 'wb') as f:
        pickle.dump((devices_train_list, val_loader, test_loader), f)

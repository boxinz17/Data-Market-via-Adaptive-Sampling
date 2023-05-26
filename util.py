import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

loss_CE_fn = nn.CrossEntropyLoss()
softmax_model = nn.Softmax(dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(1024, 300)
        self.fc2 = nn.Linear(300, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def model_eval(model, data_loader):
    # compute the validation accuracy
    count = 0
    right_count = 0
    for imgs, labels in data_loader:
        count += len(labels)
        with torch.no_grad():
            outs = softmax_model(model(imgs))
            outs = torch.argmax(outs, dim=1)
                
            # Update correct prediction numbers
            right_count += (outs == labels).sum().item()
    
    return right_count / count

def model_eval2(model, data_loader):
    # compute the validation loss
    count = 0
    loss = 0.0
    for imgs, labels in data_loader:
        count += len(labels)
        with torch.no_grad():
            outs = model(imgs)
            loss += loss_CE_fn(outs, labels)
    
    return loss / count

def OMD_solver(p, u, lr, alpha):
    n = len(p)
    log_p_new = np.log(p) + lr * u
    log_p_new_sorted = np.sort(log_p_new)
    p_new_sorted = np.exp(log_p_new_sorted)

    i_star = 0
    for i in range(n, 0, -1):
        if log_p_new_sorted[i-1] + np.log(1 - alpha * (i-1) / n) <= np.log(alpha/n) + np.log(p_new_sorted[i-1:].sum()):
            i_star = i+1
            break
    
    ss = p_new_sorted[i_star-1:].sum()
    ss_log = np.log(ss)
    p_new_argsort = np.argsort(log_p_new)
    p_hat = np.zeros(n)
    for i in range(n):
        if i+1 < i_star:
            p_hat[p_new_argsort[i]] = alpha / n
        else:
            p_hat[p_new_argsort[i]] = np.exp( log_p_new[p_new_argsort[i]] + np.log(1 - alpha * (i_star-1) / n) - ss_log )
    
    return p_hat / p_hat.sum()

def train_FedAvg_uniform(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun):
    # number of providers
    n_providers = len(providers_train_list)
    
    # intialize the number of access for each data provider
    n_access = np.zeros(n_providers)

    # initialize the global model
    if dataset_name == 'CIFAR10':
       model = Net4()
    else:
       model = Net()
    
    util_list = []  # list storing validation lost
    test_accu_list = []  # list storing test accuracy

    start_time = time.time()
    for t in range(n_commun):
        # shrink the optimization stepsize every 200 communication rounds
        if (t+1) % 200 == 0:
            step_size_local /= 2
            step_size_global = step_size_local
    
        # compute the performance of the global model on hold-out test set
        test_accu = model_eval(model, test_loader)
        test_accu_list.append(test_accu)
    
        # compute the utility value of the global model
        util = -model_eval2(model, val_loader)
        util_list.append(util.item())
    
        # print out useful information
        print('communication round: {} | evaluation loss: {} | test accuracy: {:.3f}%'.format(t+1, -util, test_accu*100))
        print(n_access)
    
        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False)
    
        # store the current global model parameters
        global_params = []
        for f in model.parameters():
            global_params.append(f.data)
    
        updates = []  # store the local updates
        for i in chosen_providers:
            n_access[i] += 1
            local_updates = []
        
            # initialize the local model
            if dataset_name == 'CIFAR10':
                model_local = Net4()
            else:
                model_local = Net()
        
            model_local.load_state_dict(model.state_dict())
            optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
        
            # make local computation
            for e in range(n_epoch):
                for imgs, labels in providers_train_list[i]:
                    optim_local.zero_grad()
                    outs = model_local(imgs)
                    loss = loss_CE_fn(outs, labels) / len(labels)
                    loss.backward()
                    optim_local.step()
        
            # compute the local updates
            for j, f in enumerate(model_local.parameters()):
                local_updates.append((f.data-global_params[j])/step_size_local)
            updates.append(local_updates)
    
        # make updates of the global model by FedAvg
        for j, f in enumerate(model.parameters()):
            update = updates[0][j]
            for i in range(1,batch_size):
                update += updates[i][j]
            update /= batch_size
            f.data.add_(step_size_global*update)

    end_time = time.time()
    time_used = end_time - start_time

    return model, n_access, util_list, test_accu_list, time_used

def train_FedAvg_OSMD(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun, alpha, learning_rate):
    # number of providers
    n_providers = len(providers_train_list) 
    
    # intialize the number of access for each data provider
    n_access = np.zeros(n_providers)

    # initialize the global model
    if dataset_name == 'CIFAR10':
        model = Net4()
    else:
        model = Net()

    prob_sampling = np.ones(n_providers)
    prob_sampling /= prob_sampling.sum()  # initialize the sampling distribution as uniform
    u_hat = np.zeros(n_providers)
    util_list = []  # list storing validation lost
    test_accu_list = []  # list storing test accuracy
    start_time = time.time()
    for t in range(n_commun):
        # shrink the optimization stepsize every 200 communication rounds
        if (t+1) % 200 == 0:
            step_size_local /= 2
            step_size_global = step_size_local

        # compute the performance of the global model on
        # hold-out test set
        test_accu = model_eval(model, test_loader)
        test_accu_list.append(test_accu)

        # compute the utility value of the global model
        util = -model_eval2(model, val_loader)
        util_list.append(util.item())
    
        # print out useful information
        print('communication round: {} | evaluation loss: {} | test accuracy: {:.3f}%'.format(t+1, -util, test_accu*100))
        print(n_access)
        print(prob_sampling)
        print(u_hat)
    
        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=True, p=prob_sampling)
    
        # store the current global model parameters
        global_params = []
        for f in model.parameters():
            global_params.append(f.data)
    
        updates = []  # store the local updates
        u_hat = np.zeros(n_providers)  # reset the u_hat vector
        for i in chosen_providers:
            n_access[i] += 1
            local_updates = []
        
            # initialize the local model
            if dataset_name == 'CIFAR10':
                model_local = Net4()
            else:
                model_local = Net()
        
            model_local.load_state_dict(model.state_dict())
            optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
        
            # make local computation
            for e in range(n_epoch):
                for imgs, labels in providers_train_list[i]:
                    optim_local.zero_grad()
                    outs = model_local(imgs)
                    loss = loss_CE_fn(outs, labels) / len(labels)
                    loss.backward()
                    optim_local.step()
        
            # compute the local utility value
            util_new = -model_eval2(model_local, val_loader)
            u_hat[i] = (util_new - util) / (batch_size * prob_sampling[i])
        
            # compute the local updates
            for j, f in enumerate(model_local.parameters()):
                local_updates.append((f.data-global_params[j])/step_size_local)
            updates.append(local_updates)
    
        # make updates of the global model by FedAvg
        for j, f in enumerate(model.parameters()):
            update = updates[0][j]
            for i in range(1,batch_size):
                update += updates[i][j]
            update /= batch_size
            f.data.add_(step_size_global*update)
    
        prob_sampling = OMD_solver(prob_sampling, u_hat, learning_rate, alpha)

    end_time = time.time()
    time_used = end_time - start_time

    return model, n_access, util_list, test_accu_list, time_used, prob_sampling

def train_DSV(providers_train_list, val_loader, dataset_name, step_size_local, n_epoch, back_check, c_tol, max_n_perm):
    # params:
    #   back_check:  integer, the gap to check convergence of DSV
    #   c_tol: float, tolerance to decide if DSV has converged
    #   max_n_perm: integer, maximum number of permutations
    
    n_providers = len(providers_train_list)  # number of providers
    
    DSV_list = []  # List to store the DSV values
    DSV = np.zeros(n_providers)  # initilaize the Data Shapley values
    DSV_convg_measure = []  # check if DSV has converged
    
    n_access = np.zeros(n_providers)  # the number of access for each data provider

    start_time = time.time()
    t  = 0
    while True:
        # check if the number of permutations exceeds the maximum threshold
        t += 1
        if t > max_n_perm:
            break

        # shuffle all the training samples
        shuffled_seq = np.arange(n_providers)
        np.random.shuffle(shuffled_seq)

        # initialize the local model
        if dataset_name == 'CIFAR10':
            model_local = Net4()
        else:
            model_local = Net()
    
        optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
    
        v = 0.0  # intialize the value function
        for count_i, i in enumerate(shuffled_seq):
            # make local computation
            for e in range(n_epoch):
                for imgs, labels in providers_train_list[i]:
                    optim_local.zero_grad()
                    outs = model_local(imgs)
                    loss = loss_CE_fn(outs, labels) / len(labels)
                    loss.backward()
                    optim_local.step()
        
            if count_i == 0:
                # intialize the value function
                v = -model_eval2(model_local, val_loader).item()
                continue

            # compute the local utility value
            v_new = -model_eval2(model_local, val_loader).item()

            # update the DSV
            DSV[i] = ((t-1) / t) * DSV[i] + (1 / t) * (v_new - v)

            # update the value function
            v = v_new

        # append the DSV to DSV list
        DSV_list.append(DSV.copy())

        # print out the DSV
        print(DSV)

        # compute the convergence criterion of DSV
        if t > back_check:
            s = 0.0
            count = 0
            for i in range(len(DSV)):
                if abs(DSV_list[-1][i]) > 0.0:
                    count += 1
                    s += abs(DSV_list[-1][i] - DSV_list[-1-back_check][i]) / abs(DSV_list[-1][i])
            s /= count
            DSV_convg_measure.append(s)

            if s < c_tol:
               print("The permutation round: {} | convergence criterion: {:.3f}".format(t, DSV_convg_measure[-1]))
               break

        # print out useful information
        if len(DSV_convg_measure) > 0:
            print("The permutation round: {} | convergence criterion: {:.3f}".format(t, DSV_convg_measure[-1]))
        else:
            print("The permutation round: {}".format(t))

    end_time = time.time()
    time_used = end_time - start_time

    return DSV_list, DSV_convg_measure, DSV, time_used

def train_FedAvg_FedDSV(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun, c_tol, back_check):
    # params:
    #   c_tol: float, convgernece tolerance to decide if DSV has converged
    #   back_check: integer, the gap to check convergence of DSV

    n_providers = len(providers_train_list)  # number of providers
    n_access = np.zeros(n_providers)

    # initialize the global model
    if dataset_name == 'CIFAR10':
        model = Net4()
    else:
        model = Net()

    util_list = []  # list storing validation lost
    test_accu_list = []  # list storing test accuracy
    start_time = time.time()
    FedDSV = np.zeros(n_providers)  # initilaize the Federated Data Shapley values

    for t in range(n_commun):
        # shrink the optimization stepsize every 200 communication rounds
        if (t+1) % 200 == 0:
            step_size_local /= 2
            step_size_global = step_size_local
    
        # compute the performance of the global model on
        # hold-out test set
        test_accu = model_eval(model, test_loader)
        test_accu_list.append(test_accu)
    
        # compute the utility value of the global model
        util = model_eval2(model, val_loader)
        util_list.append(util.item())
    
        # print out useful information
        print('communication round: {} | evaluation loss: {} | test accuracy: {:.3f}%'.format(t+1, util, test_accu*100))
        print(n_access)
        print(FedDSV)
    
        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False)
    
        # store the current global model parameters
        global_params = []
        for f in model.parameters():
            global_params.append(f.data)
    
        updates = []  # store the local updates
        for i in chosen_providers:
            n_access[i] += 1
            local_updates = []
        
            # initialize the local model
            if dataset_name == 'CIFAR10':
                model_local = Net4()
            else:
                model_local = Net()
        
            model_local.load_state_dict(model.state_dict())
            optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
        
            # make local computation
            for e in range(n_epoch):
                for imgs, labels in providers_train_list[i]:
                    optim_local.zero_grad()
                    outs = model_local(imgs)
                    loss = loss_CE_fn(outs, labels) / len(labels)
                    loss.backward()
                    optim_local.step()
        
            # compute the local updates
            for j, f in enumerate(model_local.parameters()):
                local_updates.append((f.data-global_params[j])/step_size_local)
            updates.append(local_updates)
    
        # compute the Federated Data Shapley value of this round
        FedDSV_t = np.zeros(n_providers)
        DSV_convg_measure = []  # list of convergence criterion of DSV
        DSV_list = []  # list storing DSV
        n_perm = 0  # number of permutations
        chosen_providers_shuffled = np.arange(batch_size)
        while True:
            n_perm += 1

            # shuffle all chosen providers
            np.random.shuffle(chosen_providers_shuffled)
        
            # create a temp model with the same parameter weights as global model
            if dataset_name == 'CIFAR10':
                model_temp = Net4()
            else:
                model_temp = Net()
        
            model_temp.load_state_dict(model.state_dict())

            # initialize the value function
            U_prev = -util.item()

            # update the Fed DSV of this round
            for i in chosen_providers_shuffled:
                for j, f in enumerate(model_temp.parameters()):
                    f.data.add_(step_size_local*updates[i][j])
                U_new = -model_eval2(model_temp, val_loader).item()
                FedDSV_t[chosen_providers[i]] =  ((n_perm-1) / n_perm) * FedDSV_t[chosen_providers[i]] + (1 / n_perm) * (U_new - U_prev)
                U_prev = U_new

            # append the DSV to DSV list
            DSV_list.append(FedDSV_t.copy())

            if n_perm > back_check:
                s = 0.0
                count = 0
                for i in range(batch_size):
                    if abs(DSV_list[-1][chosen_providers[i]]) > 0.0:
                        count += 1
                        s += abs(DSV_list[-1][chosen_providers[i]] - DSV_list[-1-back_check][chosen_providers[i]]) / abs(DSV_list[-1][chosen_providers[i]])
                s /= count
                DSV_convg_measure.append(s)

                if s < c_tol:
                    print("Fed DSV of round {} converged | # permutations: {} | convergence critertion: {}".format(t+1, n_perm, s))
                    break
    
        # update the cumulative Fed DSV
        FedDSV += FedDSV_t

        # make updates of the global model by FedAvg
        for j, f in enumerate(model.parameters()):
            update = updates[0][j]
            for i in range(1,batch_size):
                update += updates[i][j]
            update /= batch_size
            f.data.add_(step_size_global*update)

    end_time = time.time()
    time_used = end_time - start_time

    return model, n_access, util_list, test_accu_list, time_used, FedDSV

def train_FedAvg_best(providers_train_list, val_loader, test_loader, dataset_name, step_size_local, step_size_global, n_epoch, batch_size, n_commun):
    # number of providers
    n_providers = len(providers_train_list)
    
    # intialize the number of access for each data provider
    n_access = np.zeros(n_providers)

    # initialize the global model
    if dataset_name == 'CIFAR10':
       model = Net4()
    else:
       model = Net()
    
    util_list = []  # list storing validation lost
    test_accu_list = []  # list storing test accuracy

    start_time = time.time()
    for t in range(n_commun):
        # shrink the optimization stepsize every 200 communication rounds
        if (t+1) % 200 == 0:
            step_size_local /= 2
            step_size_global = step_size_local
    
        # compute the performance of the global model on hold-out test set
        test_accu = model_eval(model, test_loader)
        test_accu_list.append(test_accu)
    
        # compute the utility value of the global model
        util = model_eval2(model, val_loader)
        util_list.append(util.item())
    
        # print out useful information
        print('communication round: {} | evaluation loss: {} | test accuracy: {:.3f}%'.format(t+1, util, test_accu*100))
        print(n_access)
    
        # sample providers
        chosen_providers = np.arange(10)
    
        # store the current global model parameters
        global_params = []
        for f in model.parameters():
            global_params.append(f.data)
    
        updates = []  # store the local updates
        for i in chosen_providers:
            n_access[i] += 1
            local_updates = []
        
            # initialize the local model
            if dataset_name == 'CIFAR10':
                model_local = Net4()
            else:
                model_local = Net()
        
            model_local.load_state_dict(model.state_dict())
            optim_local = torch.optim.SGD(model_local.parameters(), lr=step_size_local, momentum=0.1)
        
            # make local computation
            for e in range(n_epoch):
                for imgs, labels in providers_train_list[i]:
                    optim_local.zero_grad()
                    outs = model_local(imgs)
                    loss = loss_CE_fn(outs, labels) / len(labels)
                    loss.backward()
                    optim_local.step()
        
            # compute the local updates
            for j, f in enumerate(model_local.parameters()):
                local_updates.append((f.data-global_params[j])/step_size_local)
            updates.append(local_updates)
    
        # make updates of the global model by FedAvg
        for j, f in enumerate(model.parameters()):
            update = updates[0][j]
            for i in range(1,batch_size):
                update += updates[i][j]
            update /= batch_size
            f.data.add_(step_size_global*update)

    end_time = time.time()
    time_used = end_time - start_time

    return model, n_access, util_list, test_accu_list, time_used
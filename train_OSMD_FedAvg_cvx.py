import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)
    
    def forward(self, x):
        x = self.fc1(x.view(x.shape[0], -1))
        return x

loss_CE_fn = nn.CrossEntropyLoss()
softmax_model = nn.Softmax(dim=1)

def model_eval(model, data_loader):
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

def OMS_solver(p, u, lr, alpha):
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

with open('data/mnist/corrupted_data.pickle', 'rb') as f:
    providers_train_list, val_loader, test_loader = pickle.load(f)

n_providers = len(providers_train_list)
step_size_local = 1e-3  # step size for local updates
step_size_global = step_size_local  # step size for global updates
n_epoch = 3  # number of epochs for local computation of each round
batch_size = 5
n_commun = 1000
alpha = np.sqrt(n_providers/(batch_size*n_commun))
learning_rate = np.sqrt(np.log(n_providers*batch_size*n_commun)/(n_providers*batch_size*n_commun))  # learning rate to update sampling distribution
print('alpha: {} | learning rate: {}'.format(alpha, learning_rate))

n_access = np.zeros(n_providers)

np.random.seed(111)
model = Net()
prob_sampling = np.ones(n_providers)
prob_sampling /= prob_sampling.sum()
u_hat = np.zeros(n_providers)
for t in range(n_commun):
    # compute the utility value of the global model
    util = model_eval(model, val_loader)
    print('communication round: {} | evaluation accuracy: {:.3f}%'.format(t+1, util*100))
    
    print(prob_sampling)
    print(u_hat)
    u_hat = np.zeros(n_providers)  # reset the u_hat vector
    
    chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False, p=prob_sampling)
    
    # store the current global model parameters
    global_params = []
    for f in model.parameters():
        global_params.append(f.data)
    
    updates = []  # store the local updates
    util_news = np.zeros(batch_size)  # initialize the utilities
    for i in chosen_providers:
        n_access[i] += 1
        updates.append([])
        
        # initialize the local model
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
        util_new = model_eval(model_local, val_loader)
        print('evaluation accuracy: {}%'.format(util_new*100))
        u_hat[i] += (util_new - util) / prob_sampling[i]
        
        # compute the local updates
        for j, f in enumerate(model_local.parameters()):
            updates[-1].append((f.data-global_params[j])/step_size_local)
    
    # make updates of the global model by FedAvg
    for j, f in enumerate(model.parameters()):
        update = updates[0][j]
        for i in range(1,batch_size):
            update += updates[i][j]
        update /= batch_size
        f.data.add_(step_size_global*update)
    
    prob_sampling = OMS_solver(prob_sampling, u_hat, learning_rate, alpha)